%% ============================================================
%  SIMULACAO COMPLETA - ROBOT 2WD + ENCODER + TRIANGULACAO + EKF
%
%  Estado:
%       X = [x y theta]'
%
%  Sensores:
%       1) Encoders das rodas
%       2) Sistema de triangulacao fornecendo x,y
%
%  Ground truth:
%       Trajetoria analitica
%
%  Frequencia:
%       10 Hz
%
% =============================================================

clear;
close all;
clc;

rng(10);    % Reprodutibilidade do ruido

%% ============================================================
% 1. PARAMETROS DO ROBO
% ============================================================

r = 0.03;               % Raio da roda [m]
L = 0.26;               % Distancia entre rodas [m]

Ts = 0.1;               % Periodo de amostragem [s]
fs = 1/Ts;              % Frequencia [Hz]

Tsim = 90;              % Tempo de simulacao [s]

%% ============================================================
% 2. ENCODER
% ============================================================

PPR = 1200;             % Pulsos por volta

% Distancia percorrida por pulso
d_pulse = 2*pi*r/PPR;

% Desvio-padrao da quantizacao
sigma_quant = d_pulse/sqrt(12);

fprintf('\n');
fprintf('===============================================\n');
fprintf('PARAMETROS DO ENCODER\n');
fprintf('===============================================\n');

fprintf('Pulsos/volta             = %d\n', PPR);
fprintf('Distancia/pulso          = %.8f m\n', d_pulse);
fprintf('Distancia/pulso          = %.4f mm\n', d_pulse*1000);
fprintf('Sigma quantizacao        = %.6f mm\n', ...
    sigma_quant*1000);

%% ============================================================
% 3. VELOCIDADES DAS RODAS
% ============================================================

omega_R_deg = 300;       % [deg/s]
omega_L_deg = 270;       % [deg/s]

omega_R = deg2rad(omega_R_deg);
omega_L = deg2rad(omega_L_deg);

% Velocidades lineares
v_R = r*omega_R;
v_L = r*omega_L;

% Velocidade linear do robo
v = (v_R + v_L)/2;

% Velocidade angular do robo
omega = (v_R - v_L)/L;

% Raio da trajetoria
R = v/omega;

%% ============================================================
% 4. VETOR DE TEMPO
% ============================================================

t = 0:Ts:Tsim;
N = length(t);

%% ============================================================
% 5. GROUND TRUTH ANALITICO
% ============================================================

x_gt = R*sin(omega*t);

y_gt = R*(1-cos(omega*t));

theta_gt = omega*t;

%% ============================================================
% 6. GROUND TRUTH DOS DESLOCAMENTOS DAS RODAS
% ============================================================

% Deslocamento verdadeiro por amostra

dsR_true = v_R*Ts*ones(1,N-1);

dsL_true = v_L*Ts*ones(1,N-1);

% Pulsos verdadeiros por amostra

pulses_R_true = dsR_true/d_pulse;

pulses_L_true = dsL_true/d_pulse;

%% ============================================================
% 7. PARAMETROS DE ERRO DOS ENCODERS
% ============================================================

% IMPORTANTE:
%
% sigma_wheel representa o erro efetivo de deslocamento
% de cada roda por amostra.
%
% Ele inclui efeitos que nao aparecem na simples
% quantizacao do encoder:
%
%   - escorregamento
%   - erro do raio efetivo
%   - pequenas diferencas entre rodas
%   - erro mecanico
%   - etc.
%
% Valor inicial para simulacao:
%
%       2 mm por amostra
%
% Esse valor deve ser posteriormente determinado
% experimentalmente.

sigma_wheel = 0.002;       % [m]

%% ============================================================
% 8. SIMULACAO DOS ENCODERS
% ============================================================

% Ruido equivalente em pulsos

sigma_pulse_noise = sigma_wheel/d_pulse;

% Pulsos medidos com ruido
%
% Aqui mantemos a contagem em numeros reais para
% separar claramente o efeito do ruido.
%
% Na simulacao de um encoder digital real podemos
% posteriormente aplicar round().

pulses_R_meas = pulses_R_true + ...
                sigma_pulse_noise*randn(1,N-1);

pulses_L_meas = pulses_L_true + ...
                sigma_pulse_noise*randn(1,N-1);

% Deslocamento medido

dsR_meas = pulses_R_meas*d_pulse;

dsL_meas = pulses_L_meas*d_pulse;

%% ============================================================
% 9. ODOMETRIA PURA
% ============================================================

x_odom = zeros(1,N);
y_odom = zeros(1,N);
theta_odom = zeros(1,N);

for k = 1:N-1

    % Deslocamento medio
    ds = (dsR_meas(k) + dsL_meas(k))/2;

    % Variacao angular
    dtheta = (dsR_meas(k) - dsL_meas(k))/L;

    % Integracao no ponto medio
    theta_mid = theta_odom(k) + dtheta/2;

    x_odom(k+1) = ...
        x_odom(k) + ds*cos(theta_mid);

    y_odom(k+1) = ...
        y_odom(k) + ds*sin(theta_mid);

    theta_odom(k+1) = ...
        theta_odom(k) + dtheta;

end

%% ============================================================
% 10. TRIANGULACAO
% ============================================================

% Desvio-padrao da triangulacao

sigma_tri = 0.10;          % [m]

% Medidas de posicao

x_tri = x_gt + sigma_tri*randn(size(x_gt));

y_tri = y_gt + sigma_tri*randn(size(y_gt));

%% ============================================================
% 11. MATRIZ DE MEDICAO R
% ============================================================

Rk = [sigma_tri^2   0;
      0             sigma_tri^2];

%% ============================================================
% 12. EKF - INICIALIZACAO
% ============================================================

% Estado inicial

x_ekf = zeros(3,N);

% Inicialmente assumimos que conhecemos a posicao inicial,
% mas nao necessariamente a orientacao.

x_ekf(:,1) = [0;
              0;
              0];

% Covariancia inicial

P = diag([ ...
    0.01^2, ...        % x
    0.01^2, ...        % y
    deg2rad(5)^2]);    % theta

%% ============================================================
% 13. MATRIZ H
% ============================================================

H = [1 0 0;
     0 1 0];

%% ============================================================
% 14. EKF
% ============================================================

for k = 1:N-1

    %% --------------------------------------------------------
    % PREDICAO
    % ---------------------------------------------------------

    dsR = dsR_meas(k);
    dsL = dsL_meas(k);

    ds = (dsR + dsL)/2;

    dtheta = (dsR - dsL)/L;

    theta_old = x_ekf(3,k);

    theta_mid = theta_old + dtheta/2;

    % Estado previsto

    x_pred = x_ekf(1,k) + ...
             ds*cos(theta_mid);

    y_pred = x_ekf(2,k) + ...
             ds*sin(theta_mid);

    theta_pred = theta_old + dtheta;

    x_pred_state = [ ...
        x_pred;
        y_pred;
        theta_pred];

    %% --------------------------------------------------------
    % JACOBIANA F
    % ---------------------------------------------------------

    F = [ ...
        1, 0, -ds*sin(theta_mid);
        0, 1,  ds*cos(theta_mid);
        0, 0,  1];

    %% --------------------------------------------------------
    % MATRIZ G - RUIDO DOS ENCODERS
    % ---------------------------------------------------------

    G = [0.5*cos(theta_mid) ...
        - ds/(2*L)*sin(theta_mid), ...
        0.5*cos(theta_mid) ...
        + ds/(2*L)*sin(theta_mid);
        0.5*sin(theta_mid) ...
        + ds/(2*L)*cos(theta_mid), ...
        0.5*sin(theta_mid) ...
        - ds/(2*L)*cos(theta_mid);
        1/L, ...
        -1/L];

    %% --------------------------------------------------------
    % MATRIZ DE COVARIANCIA DOS ENCODERS
    % ---------------------------------------------------------

    Sigma_encoder = [ ...
        sigma_wheel^2, 0;
        0, sigma_wheel^2];

    %% --------------------------------------------------------
    % MATRIZ Q
    % ---------------------------------------------------------

    Q = G*Sigma_encoder*G';

    %% --------------------------------------------------------
    % PREDICAO DA COVARIANCIA
    % ---------------------------------------------------------

    P_pred = F*P*F' + Q;

    %% --------------------------------------------------------
    % CORRECAO COM TRIANGULACAO
    % ---------------------------------------------------------

    z = [x_tri(k+1);
         y_tri(k+1)];

    % Medicao prevista

    z_pred = H*x_pred_state;

    % Inovacao

    innovation = z - z_pred;

    % Covariancia da inovacao

    S = H*P_pred*H' + Rk;

    % Ganho de Kalman

    K = P_pred*H'/S;

    % Atualizacao do estado

    x_update = x_pred_state + ...
               K*innovation;

    %% --------------------------------------------------------
    % ATUALIZACAO DA COVARIANCIA
    % ---------------------------------------------------------

    I = eye(3);

    % Forma de Joseph:
    %
    % numericamente mais robusta

    P = (I-K*H)*P_pred*(I-K*H)' + K*Rk*K';

    %% --------------------------------------------------------
    % NORMALIZAR THETA
    % ---------------------------------------------------------

    x_update(3) = atan2( ...
        sin(x_update(3)), ...
        cos(x_update(3)));

    %% --------------------------------------------------------
    % SALVAR ESTADO
    % ---------------------------------------------------------

    x_ekf(:,k+1) = x_update;

end

%% ============================================================
% 15. NORMALIZAR ANGULOS PARA COMPARACAO
% ============================================================

theta_odom_wrapped = atan2( ...
    sin(theta_odom), ...
    cos(theta_odom));

theta_ekf = x_ekf(3,:);

%% ============================================================
% 16. ERROS
% ============================================================

error_odom_x = x_odom - x_gt;
error_odom_y = y_odom - y_gt;

error_ekf_x = x_ekf(1,:) - x_gt;
error_ekf_y = x_ekf(2,:) - y_gt;

error_odom_theta = ...
    atan2(sin(theta_odom-theta_gt), ...
          cos(theta_odom-theta_gt));

error_ekf_theta = ...
    atan2(sin(theta_ekf-theta_gt), ...
          cos(theta_ekf-theta_gt));

%% ============================================================
% 17. RMSE
% ============================================================

RMSE_odom_x = sqrt(mean(error_odom_x.^2));
RMSE_odom_y = sqrt(mean(error_odom_y.^2));

RMSE_ekf_x = sqrt(mean(error_ekf_x.^2));
RMSE_ekf_y = sqrt(mean(error_ekf_y.^2));

RMSE_odom_theta = ...
    rad2deg(sqrt(mean(error_odom_theta.^2)));

RMSE_ekf_theta = ...
    rad2deg(sqrt(mean(error_ekf_theta.^2)));

fprintf('\n');
fprintf('===============================================\n');
fprintf('RESULTADOS\n');
fprintf('===============================================\n');

fprintf('\nRMSE POSICAO\n');

fprintf('Odometria - x = %.4f m\n', RMSE_odom_x);
fprintf('Odometria - y = %.4f m\n', RMSE_odom_y);

fprintf('EKF       - x = %.4f m\n', RMSE_ekf_x);
fprintf('EKF       - y = %.4f m\n', RMSE_ekf_y);

fprintf('\nRMSE ORIENTACAO\n');

fprintf('Odometria = %.4f graus\n', ...
    RMSE_odom_theta);

fprintf('EKF       = %.4f graus\n', ...
    RMSE_ekf_theta);

fprintf('\n===============================================\n');

%% ============================================================
% 18. GRAFICO - TRAJETORIA
% ============================================================

figure('Name','Comparacao das trajetorias');

plot(x_gt, y_gt, ...
    'LineWidth',2.5);

hold on;

plot(x_odom, y_odom, ...
    '--', ...
    'LineWidth',1.5);

plot(x_ekf(1,:), x_ekf(2,:), ...
    '-.', ...
    'LineWidth',1.8);

plot(x_tri, y_tri, ...
    '.', ...
    'MarkerSize',8);

plot(x_gt(1), y_gt(1), ...
    'o', ...
    'MarkerSize',8, ...
    'LineWidth',2);

grid on;
axis equal;

xlabel('x [m]');
ylabel('y [m]');

title('Ground Truth x Odometria x EKF x Triangulacao');

legend( ...
    'Ground Truth', ...
    'Odometria', ...
    'EKF', ...
    'Triangulacao', ...
    'Inicio', ...
    'Location','best');

%% ============================================================
% 19. GRAFICO - X
% ============================================================

figure('Name','Posicao X');

plot(t,x_gt, ...
    'LineWidth',2);

hold on;

plot(t,x_odom, ...
    '--', ...
    'LineWidth',1.5);

plot(t,x_ekf(1,:), ...
    '-.', ...
    'LineWidth',1.8);

plot(t,x_tri, ...
    '.', ...
    'MarkerSize',7);

grid on;

xlabel('Tempo [s]');
ylabel('x [m]');

title('Estimativa da Posicao X');

legend( ...
    'Ground Truth', ...
    'Odometria', ...
    'EKF', ...
    'Triangulacao', ...
    'Location','best');

%% ============================================================
% 20. GRAFICO - Y
% ============================================================

figure('Name','Posicao Y');

plot(t,y_gt, ...
    'LineWidth',2);

hold on;

plot(t,y_odom, ...
    '--', ...
    'LineWidth',1.5);

plot(t,x_ekf(2,:), ...
    '-.', ...
    'LineWidth',1.8);

plot(t,y_tri, ...
    '.', ...
    'MarkerSize',7);

grid on;

xlabel('Tempo [s]');
ylabel('y [m]');

title('Estimativa da Posicao Y');

legend( ...
    'Ground Truth', ...
    'Odometria', ...
    'EKF', ...
    'Triangulacao', ...
    'Location','best');

%% ============================================================
% 21. GRAFICO - THETA
% ============================================================

figure('Name','Orientacao');

plot(t,rad2deg(theta_gt), ...
    'LineWidth',2);

hold on;

plot(t,rad2deg(theta_odom_wrapped), ...
    '--', ...
    'LineWidth',1.5);

plot(t,rad2deg(theta_ekf), ...
    '-.', ...
    'LineWidth',1.8);

grid on;

xlabel('Tempo [s]');
ylabel('\theta [graus]');

title('Estimativa da Orientacao');

legend( ...
    'Ground Truth', ...
    'Odometria', ...
    'EKF', ...
    'Location','best');

%% ============================================================
% 22. GRAFICO DOS ERROS DE POSICAO
% ============================================================

figure('Name','Erro de Posicao');

plot(t,error_odom_x, ...
    '--', ...
    'LineWidth',1.5);

hold on;

plot(t,error_ekf_x, ...
    'LineWidth',1.5);

plot(t,error_odom_y, ...
    '--', ...
    'LineWidth',1.5);

plot(t,error_ekf_y, ...
    'LineWidth',1.5);

grid on;

xlabel('Tempo [s]');
ylabel('Erro [m]');

title('Erro de Posicao');

legend( ...
    'Erro Odometria X', ...
    'Erro EKF X', ...
    'Erro Odometria Y', ...
    'Erro EKF Y', ...
    'Location','best');

%% ============================================================
% 23. GRAFICO DO ERRO ANGULAR
% ============================================================

figure('Name','Erro Angular');

plot(t,rad2deg(error_odom_theta), ...
    '--', ...
    'LineWidth',1.5);

hold on;

plot(t,rad2deg(error_ekf_theta), ...
    'LineWidth',1.8);

grid on;

xlabel('Tempo [s]');
ylabel('Erro angular [graus]');

title('Erro de Orientacao');

legend( ...
    'Odometria', ...
    'EKF', ...
    'Location','best');

%% ============================================================
% 24. COVARIANCIAS DO EKF
% ============================================================

sigma_x_ekf = zeros(1,N);
sigma_y_ekf = zeros(1,N);
sigma_theta_ekf = zeros(1,N);

% Reexecutamos a propagacao das covariancias para armazenar
% os desvios-padrao.

P = diag([ ...
    0.01^2, ...
    0.01^2, ...
    deg2rad(5)^2]);

for k = 1:N-1

    dsR = dsR_meas(k);
    dsL = dsL_meas(k);

    ds = (dsR + dsL)/2;

    dtheta = (dsR-dsL)/L;

    theta_mid = x_ekf(3,k) + dtheta/2;

    F = [1 0 -ds*sin(theta_mid);
        0 1  ds*cos(theta_mid);
        0 0  1];

    G = [0.5*cos(theta_mid) - ds/(2*L)*sin(theta_mid), 0.5*cos(theta_mid) + ds/(2*L)*sin(theta_mid);
         0.5*sin(theta_mid) + ds/(2*L)*cos(theta_mid), 0.5*sin(theta_mid) - ds/(2*L)*cos(theta_mid);
         1/L, -1/L];

    Q = G*Sigma_encoder*G';

    P_pred = F*P*F' + Q;

    S = H*P_pred*H' + Rk;

    K = P_pred*H'/S;

    I = eye(3);

    P = (I-K*H)*P_pred*(I-K*H)' + K*Rk*K';

    sigma_x_ekf(k+1) = sqrt(P(1,1));
    sigma_y_ekf(k+1) = sqrt(P(2,2));
    sigma_theta_ekf(k+1) = sqrt(P(3,3));

end

%% ============================================================
% 25. COVARIANCIAS
% ============================================================

figure('Name','Incerteza EKF');

plot(t,sigma_x_ekf, ...
    'LineWidth',1.5);

hold on;

plot(t,sigma_y_ekf, ...
    'LineWidth',1.5);

plot(t,rad2deg(sigma_theta_ekf), ...
    'LineWidth',1.5);

grid on;

xlabel('Tempo [s]');
ylabel('Desvio padrao');

title('Incerteza estimada pelo EKF');

legend( ...
    '\sigma_x [m]', ...
    '\sigma_y [m]', ...
    '\sigma_\theta [graus]', ...
    'Location','best');

%% ============================================================
% 26. SALVAR DADOS
% ============================================================

simulation.t = t;

simulation.ground_truth.x = x_gt;
simulation.ground_truth.y = y_gt;
simulation.ground_truth.theta = theta_gt;

simulation.odom.x = x_odom;
simulation.odom.y = y_odom;
simulation.odom.theta = theta_odom;

simulation.triangulation.x = x_tri;
simulation.triangulation.y = y_tri;

simulation.ekf.x = x_ekf;

simulation.encoder.pulses_R = pulses_R_meas;
simulation.encoder.pulses_L = pulses_L_meas;

simulation.parameters.r = r;
simulation.parameters.L = L;
simulation.parameters.PPR = PPR;
simulation.parameters.Ts = Ts;
simulation.parameters.sigma_wheel = sigma_wheel;
simulation.parameters.sigma_tri = sigma_tri;

save('simulation_2WD_EKF.mat','simulation');

fprintf('\nDados salvos em:\n');
fprintf('simulation_2WD_EKF.mat\n');