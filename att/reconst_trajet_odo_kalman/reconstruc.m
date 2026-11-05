%% ================================================================
%  RECONSTRUCAO DE TRAJETORIA - VEICULO 2WD
%
%  Dados:
%  Coluna 1 -> timestamp [ms]
%  Coluna 2 -> contador acumulado do encoder esquerdo
%  Coluna 3 -> contador acumulado do encoder direito
%
%  Saida:
%  x       -> posicao X [m]
%  y       -> posicao Y [m]
%  theta   -> orientacao [rad]
%  v       -> velocidade linear [m/s]
%  omega   -> velocidade angular [rad/s]
%
% ================================================================

%%clear;
clc;
close all;

%% ================================================================
% PARAMETROS DO VEICULO
% ================================================================

r = 0.03425;       % Raio da roda [m]
L = 0.229;       % Distancia entre as rodas [m]

CPR = 2280;     % Contagens por revolucao da roda
                % ALTERE PARA O SEU ENCODER

% Se o encoder possuir reducao mecanica, CPR deve representar
% o numero de contagens correspondentes a UMA VOLTA DA RODA.

%% ================================================================
% LEITURA DOS DADOS
% ================================================================

% Arquivo contendo:
%
% timestamp    encoder_esq    encoder_dir
%
% Exemplo:
% 0            0              0
% 10           15             16
% 20           31             33
% ...

%dados = readmatrix('odometria.csv');
%dados = readmatrix('dados.txt')
% Extrai as colunas

%dados = pcntLog;
timestamp = dados(945:end,1);     % [ms]
encoder_L = dados(945:end,2);     % contagem esquerda
encoder_R = -dados(945:end,3);     % contagem direita

%% ================================================================
% CONVERSAO DO TEMPO
% ================================================================

% milissegundos -> segundos

t = timestamp / 1000;

%% ================================================================
% NUMERO DE AMOSTRAS
% ================================================================

N = length(t);

%% ================================================================
% VETORES DE SAIDA
% ================================================================

x = zeros(N,1);
y = zeros(N,1);

theta = zeros(N,1);

v = zeros(N,1);
omega = zeros(N,1);

%% ================================================================
% DESLOCAMENTO INICIAL DOS ENCODERS
% ================================================================

% Deslocamento acumulado de cada roda

sL = zeros(N,1);
sR = zeros(N,1);

%% ================================================================
% ODOMETRIA
% ================================================================

for k = 2:N

    % ------------------------------------------------------------
    % Intervalo de tempo
    % ------------------------------------------------------------

    dt = t(k) - t(k-1);

    % ------------------------------------------------------------
    % Incremento dos encoders
    % ------------------------------------------------------------

    dCount_L = encoder_L(k) - encoder_L(k-1);
    dCount_R = encoder_R(k) - encoder_R(k-1);

    % ------------------------------------------------------------
    % Distancia percorrida por cada roda
    % ------------------------------------------------------------

    dsL = 2*pi*r*dCount_L/CPR;
    dsR = 2*pi*r*dCount_R/CPR;

    % ------------------------------------------------------------
    % Distancia media percorrida pelo centro do veiculo
    % ------------------------------------------------------------

    ds = (dsR + dsL)/2;

    % ------------------------------------------------------------
    % Variacao angular do veiculo
    % ------------------------------------------------------------

    dtheta = (dsR - dsL)/L;

    % ------------------------------------------------------------
    % Atualiza orientacao
    % ------------------------------------------------------------

    theta(k) = theta(k-1) + dtheta;

    % ------------------------------------------------------------
    % Atualiza posicao
    %
    % Usando a orientacao media durante o intervalo
    % ------------------------------------------------------------

    theta_mid = theta(k-1) + dtheta/2;

    x(k) = x(k-1) + ds*cos(theta_mid);

    y(k) = y(k-1) + ds*sin(theta_mid);

    % ------------------------------------------------------------
    % Velocidade linear
    % ------------------------------------------------------------

    v(k) = ds/dt;

    % ------------------------------------------------------------
    % Velocidade angular
    % ------------------------------------------------------------

    omega(k) = dtheta/dt;

    % ------------------------------------------------------------
    % Deslocamento acumulado de cada roda
    % ------------------------------------------------------------

    sL(k) = sL(k-1) + dsL;
    sR(k) = sR(k-1) + dsR;

end

%% ================================================================
% GRAFICO DA TRAJETORIA
% ================================================================

figure;

plot(x, y, 'LineWidth', 2);

grid on;
axis equal;

xlabel('X [m]');
ylabel('Y [m]');

title('Trajetoria reconstruida por odometria');

%% ================================================================
% TRAJETORIA COM INICIO E FIM
% ================================================================

figure;

plot(x, y, 'LineWidth', 2);
hold on;

plot(x(1), y(1), 'o', 'MarkerSize', 8);
plot(x(end), y(end), 'x', 'MarkerSize', 10);

grid on;
axis equal;

xlabel('X [m]');
ylabel('Y [m]');

title('Trajetoria do veiculo');

legend('Trajetoria','Inicio','Fim');

%% ================================================================
% ORIENTACAO
% ================================================================

figure;

plot(t, theta, 'LineWidth', 2);

grid on;

xlabel('Tempo [s]');
ylabel('\theta [rad]');

title('Orientacao do veiculo');

%% ================================================================
% ORIENTACAO EM GRAUS
% ================================================================

figure;

plot(t, rad2deg(theta), 'LineWidth', 2);

grid on;

xlabel('Tempo [s]');
ylabel('\theta [graus]');

title('Orientacao do veiculo');

%% ================================================================
% VELOCIDADE LINEAR
% ================================================================

figure;

plot(t, v, 'LineWidth', 2);

grid on;

xlabel('Tempo [s]');
ylabel('v [m/s]');

title('Velocidade linear');

%% ================================================================
% VELOCIDADE ANGULAR
% ================================================================

figure;

plot(t, omega, 'LineWidth', 2);

grid on;

xlabel('Tempo [s]');
ylabel('\omega [rad/s]');

title('Velocidade angular');

%% ================================================================
% DISTANCIA PERCORRIDA POR CADA RODA
% ================================================================

figure;

plot(t, sL, 'LineWidth', 2);
hold on;

plot(t, sR, 'LineWidth', 2);

grid on;

xlabel('Tempo [s]');
ylabel('Distancia [m]');

title('Distancia percorrida pelas rodas');

legend('Esquerda','Direita');

%% ================================================================
% RESULTADO FINAL
% ================================================================

fprintf('\n');
fprintf('=============================================\n');
fprintf(' RESULTADO DA ODOMETRIA\n');
fprintf('=============================================\n');

fprintf('Posicao final X       = %.3f m\n', x(end));
fprintf('Posicao final Y       = %.3f m\n', y(end));

fprintf('Orientacao final      = %.3f rad\n', theta(end));
fprintf('Orientacao final      = %.3f graus\n', rad2deg(theta(end)));

fprintf('Distancia roda esquerda = %.3f m\n', sL(end));
fprintf('Distancia roda direita  = %.3f m\n', sR(end));

fprintf('=============================================\n');

