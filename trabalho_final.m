clear all;
close all;
clc;

%% 1. Definição do Ambiente e Parâmetros da Simulação

% Parâmetros de tempo da simulação
T = 0.05;                   % Período de amostragem (s) 
t_final = 50;               % Tempo total da simulação (s) 
t = 0:T:t_final;            % Vetor de tempo

% Frequências de atualização dos sensores
f_pred = 1/T;               % Frequência da odometria (Hz)
f_corr = 5;                 % Frequência das medições UWB (Hz)
% Razão entre as frequências para controle da atualização do EKF
update_ratio = f_pred / f_corr; 

% Posição das âncoras UWB no plano (x, y, z)
% Coordenadas baseadas na simulação do artigo 
anchors = [0 0 1;  % a1
           0 5 1;  % a2
           5 0 1]';% a3
num_anchors = size(anchors, 2);
fprintf('Ambiente configurado com %d âncoras.\n', num_anchors);

%% 2. Configuração do Robô e da Trajetória Real (Ground Truth)

% Parâmetros físicos do robô
l = 0.65 / 2; % Metade da distância da baseline (m). Baseline de 0.65m [cite: 378]
z_c = 0.5;    % Altura constante das tags no robô (m) [cite: 119]

% Estado inicial real do robô [x, y, theta]'
x_true = [2.5; 0; 0]; % [cite: 275]

% Entradas de controle (velocidades) para gerar a trajetória
v_true = 0.3;     % Velocidade linear constante (m/s) 
w_true = 7.5 * (pi/180); % Velocidade angular constante (rad/s) 

% Armazenamento da trajetória real (ground truth)
x_hist_true = zeros(3, length(t));
x_hist_true(:,1) = x_true;

% Geração da trajetória real (cinemática do robô)
for k = 2:length(t)
    theta = x_hist_true(3, k-1);
    % Modelo cinemático de Runge-Kutta de 2ª ordem
    x_hist_true(1, k) = x_hist_true(1, k-1) + v_true * T * cos(theta + w_true*T/2);
    x_hist_true(2, k) = x_hist_true(2, k-1) + v_true * T * sin(theta + w_true*T/2);
    x_hist_true(3, k) = x_hist_true(3, k-1) + w_true * T;
    % Normalizar o ângulo entre -pi e pi
    x_hist_true(3, k) = atan2(sin(x_hist_true(3, k)), cos(x_hist_true(3, k)));
end

fprintf('Trajetória real (ground truth) gerada com sucesso.\n');

%% 3. Modelo dos Sensores e Geração de Medições com Ruído
% Definição dos ruídos dos sensores
% Desvio padrão do ruído da odometria
sigma_v = 0.02;  % Ruído na velocidade linear (m/s)
sigma_w = 0.05;  % Ruído na velocidade angular (rad/s)

ruido_medicao = 0.0025;
sigma_uwb = sqrt(ruido_medicao); 

% Geração das medições ruidosas da odometria
input_noisy = [v_true + sigma_v * randn(1, length(t));
               w_true + sigma_w * randn(1, length(t))];

% Geração das medições ruidosas de distância UWB
z_hist = zeros(2 * num_anchors, length(t));
for k = 1:length(t)
    xt = x_hist_true(1, k);
    yt = x_hist_true(2, k);
    thetat = x_hist_true(3, k);
    
    % Posição real das tags no instante k
    pf_true = [xt + l*cos(thetat); yt + l*sin(thetat); z_c];
    pr_true = [xt - l*cos(thetat); yt - l*sin(thetat); z_c];

    % Calcular distâncias reais e adicionar ruído
    for i = 1:num_anchors
        dist_f = norm(pf_true - anchors(:,i)) + sigma_uwb * randn;
        dist_r = norm(pr_true - anchors(:,i)) + sigma_uwb * randn;
        z_hist(2*i-1, k) = dist_f;
        z_hist(2*i, k)   = dist_r;
    end
end
fprintf('Medições simuladas de odometria e UWB geradas com ruído.\n');

%% 4. Implementação do Filtro BC-EKF

% Estado inicial estimado e covariância inicial
x_est = [2.5; 0; 0]; % Inicia com o valor real 
P = diag([0.1, 0.1, 0.1]); % Incerteza inicial

% Matrizes de Covariância de Ruído do EKF
% Ruído do processo (Q) - Incerteza do modelo de movimento
% Baseado nos ruídos de velocidade e odometria
Q = diag([sigma_v^2, sigma_v^2, sigma_w^2]) * T; 

Q = diag([0.0001, 0.0001, 0.0001]); 

% Ruído da medição (R) - Incerteza das medições UWB
R = diag(repelem(sigma_uwb^2, 2 * num_anchors)); 

% Armazenamento do histórico de estados estimados
x_hist_est = zeros(3, length(t));
x_hist_est(:,1) = x_est;

fprintf('Iniciando o loop de filtragem do BC-EKF...\n');
% ====================== LOOP PRINCIPAL DA SIMULAÇÃO ======================
for k = 2:length(t)
    
    % --- ETAPA DE PREDIÇÃO (ALTA FREQUÊNCIA) ---
  
    % Extrai odometria ruidosa como entrada de controle
    v_k = input_noisy(1, k);
    w_k = input_noisy(2, k);
    
    theta_prev = x_est(3);
    
    % Predição do estado usando o modelo cinemático (Equação 5 e 11)
    x_pred = x_est + [v_k * T * cos(theta_prev + w_k*T/2);
                      v_k * T * sin(theta_prev + w_k*T/2);
                      w_k * T];
    x_pred(3) = atan2(sin(x_pred(3)), cos(x_pred(3))); % Normaliza ângulo

    % Jacobiano do modelo de predição (A_k) (Equação 6)
    A_k = [1 0 -v_k*T*sin(theta_prev + w_k*T/2);
           0 1  v_k*T*cos(theta_prev + w_k*T/2);
           0 0  1];
           
    % Predição da covariância do erro (Equação 12)
    P_pred = A_k * P * A_k' + Q;
    
    % --- ETAPA DE CORREÇÃO (BAIXA FREQUÊNCIA) ---

    % Verifica se é um passo de tempo para correção UWB
    if mod(k, update_ratio) == 0
        
        % Posições estimadas das tags baseadas no estado predito (Equação 8)
        xp = x_pred(1); yp = x_pred(2); theta_p = x_pred(3);
        pf_pred = [xp + l*cos(theta_p); yp + l*sin(theta_p); z_c];
        pr_pred = [xp - l*cos(theta_p); yp - l*sin(theta_p); z_c];
        
        % Vetor de medição esperada h(x_pred, 0) (Equação 7)
        h_pred = zeros(2 * num_anchors, 1);
        for i = 1:num_anchors
            h_pred(2*i-1) = norm(pf_pred - anchors(:,i));
            h_pred(2*i)   = norm(pr_pred - anchors(:,i));
        end

        % Jacobiano do modelo de medição (H_k) (Equação 9 e 10)

        H_k = zeros(2 * num_anchors, 3);
        for i = 1:num_anchors
            % Derivadas para a tag frontal (f)
            D_fi = h_pred(2*i-1);
            C_fi = -(pf_pred(1)-anchors(1,i))*l*sin(theta_p) + (pf_pred(2)-anchors(2,i))*l*cos(theta_p);
            H_k(2*i-1, :) = [(pf_pred(1)-anchors(1,i))/D_fi, (pf_pred(2)-anchors(2,i))/D_fi, C_fi/D_fi];
            
            % Derivadas para a tag traseira (r)
            D_ri = h_pred(2*i);
            C_ri = (pr_pred(1)-anchors(1,i))*l*sin(theta_p) - (pr_pred(2)-anchors(2,i))*l*cos(theta_p);
            H_k(2*i, :) = [(pr_pred(1)-anchors(1,i))/D_ri, (pr_pred(2)-anchors(2,i))/D_ri, C_ri/D_ri];
        end
        
        % Cálculo do Ganho de Kalman (K_k) (Equação 13)
        K_k = P_pred * H_k' / (H_k * P_pred * H_k' + R);
        
        % Vetor de medição real no instante k
        z_k = z_hist(:,k);
        
        % Correção do estado (x_k) (Equação 15)
        x_est = x_pred + K_k * (z_k - h_pred);
        x_est(3) = atan2(sin(x_est(3)), cos(x_est(3))); % Normaliza ângulo
        
        % Correção da covariância (P_k) (Equação 14)
        P = (eye(3) - K_k * H_k) * P_pred;
        
    else % Se não houver medição UWB, a predição se torna a estimativa
        x_est = x_pred;
        P = P_pred;
    end
    
    % Armazena o estado estimado
    x_hist_est(:, k) = x_est;
end

fprintf('Simulação concluída.\n');

%% 5. Visualização dos Resultados

fprintf('Gerando gráficos de resultados...\n');

% Gráfico 1: Comparação de Trajetórias
figure('Name', 'Comparação de Trajetórias');
hold on; grid on; axis equal;
plot(x_hist_true(1,:), x_hist_true(2,:), 'k', 'LineWidth', 2);
plot(x_hist_est(1,:), x_hist_est(2,:), 'b--', 'LineWidth', 1.5);
plot(anchors(1,:), anchors(2,:), 'r*', 'MarkerSize', 10, 'LineWidth', 2);
title('Trajetória do Robô: Real vs. Estimada pelo BC-EKF');
xlabel('Posição X (m)');
ylabel('Posição Y (m)');
legend('Trajetória Real (Ground Truth)', 'Trajetória Estimada (BC-EKF)', 'Âncoras UWB');
hold off;

% Gráfico 2: Erro de Posição e Orientação ao Longo do Tempo
% Cálculo do erro
error = x_hist_true - x_hist_est;
error(3,:) = atan2(sin(error(3,:)), cos(error(3,:))); % Normaliza erro angular
pos_error = vecnorm(error(1:2,:)); % Erro de posição Euclidiano
heading_error_deg = abs(error(3,:)) * (180/pi); % Erro de orientação em graus

figure('Name', 'Erro de Estimação ao Longo do Tempo');
subplot(2,1,1);
plot(t, pos_error, 'LineWidth', 1.5);
grid on;
title('Erro de Posição Euclidiano');
xlabel('Tempo (s)');
ylabel('Erro (m)');
rmse_pos = sqrt(mean(pos_error.^2));
legend(sprintf('RMSE = %.4f m', rmse_pos));

subplot(2,1,2);
plot(t, heading_error_deg, 'LineWidth', 1.5);
grid on;
title('Erro Absoluto de Orientação');
xlabel('Tempo (s)');
ylabel('Erro (graus)');
rmse_heading = sqrt(mean(heading_error_deg.^2));
legend(sprintf('RMSE = %.4f graus', rmse_heading));

fprintf('RMSE Posição: %.4f m\n', rmse_pos);
fprintf('RMSE Orientação: %.4f graus\n', rmse_heading);
fprintf('Processo finalizado.\n');