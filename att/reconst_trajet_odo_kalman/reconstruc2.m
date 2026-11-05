%% ================================================================
%  RECONSTRUCAO DE TRAJETORIA - VEICULO 2WD
%
%  ENTRADA:
%  Coluna 1 -> tempo [s]
%  Coluna 2 -> velocidade angular motor/roda esquerda [rad/s]
%  Coluna 3 -> velocidade angular motor/roda direita [rad/s]
%
%  SAIDA:
%  x       -> posicao X [m]
%  y       -> posicao Y [m]
%  theta   -> orientacao [rad]
%  v       -> velocidade linear [m/s]
%  omega   -> velocidade angular do veiculo [rad/s]
%
% ================================================================

%clear;
clc;
close all;

%% ================================================================
% PARAMETROS DO VEICULO
% ================================================================

r = 0.03425;       % Raio da roda [m]
L = 0.229;       % Distancia entre as rodas [m]

CPR = 2280;     % Contagens por revolucao da roda

dt = 0.250;     % Periodo de amostragem [s]

%% ================================================================
% LEITURA DOS DADOS
% ================================================================

% Arquivo:
%
% tempo     omega_L     omega_R
%
% Exemplo:
%
% 0.00      10.0        10.0
% 0.25      10.0        10.0
% 0.50      10.0        12.0
% 0.75       8.0        12.0
% ...

ddd_Dados= readmatrix('pcntLog.csv');

%% ================================================================
% EXTRACAO DOS DADOS
% ================================================================
ddaa = diff(ddd_Dados(950:end,:))
omega_L = ddaa(:,2);
omega_R = -ddaa(:,3);
t       = 0:(length(omega_L)-1);
t       = t'*0.25;
N = length(omega_R);

%% ================================================================
% VETORES DE SAIDA
% ================================================================

x = zeros(N,1);
y = zeros(N,1);
theta = zeros(N,1);

v = zeros(N,1);
omega = zeros(N,1);

%% ================================================================
% ODOMETRIA
% ================================================================

for k = 2:N

    % ------------------------------------------------------------
    % Intervalo de tempo
    % ------------------------------------------------------------

    dt_k = t(k) - t(k-1);

    % ------------------------------------------------------------
    % Velocidade linear de cada roda
    % ------------------------------------------------------------

    vL = r * omega_L(k-1);
    vR = r * omega_R(k-1);

    % ------------------------------------------------------------
    % Velocidade linear do centro do veiculo
    % ------------------------------------------------------------

    v(k) = (vR + vL)/2;

    % ------------------------------------------------------------
    % Velocidade angular do veiculo
    % ------------------------------------------------------------

    omega(k) = (vR - vL)/L;

    % ------------------------------------------------------------
    % Variacao angular durante o intervalo
    % ------------------------------------------------------------

    dtheta = omega(k) * dt_k;

    % ------------------------------------------------------------
    % Orientacao media durante o intervalo
    % ------------------------------------------------------------

    theta_mid = theta(k-1) + dtheta/2;

    % ------------------------------------------------------------
    % Deslocamento do veiculo
    % ------------------------------------------------------------

    ds = v(k) * dt_k;

    % ------------------------------------------------------------
    % Atualizacao da posicao
    % ------------------------------------------------------------

    x(k) = x(k-1) + ds*cos(theta_mid);

    y(k) = y(k-1) + ds*sin(theta_mid);

    % ------------------------------------------------------------
    % Atualizacao da orientacao
    % ------------------------------------------------------------

    theta(k) = theta(k-1) + dtheta;

end

%% ================================================================
% TRAJETORIA
% ================================================================

figure;

plot(x, y, 'LineWidth', 2);

grid on;
axis equal;

xlabel('X [m]');
ylabel('Y [m]');

title('Trajetoria reconstruida - 2WD');

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

title('Velocidade linear do veiculo');

%% ================================================================
% VELOCIDADE ANGULAR
% ================================================================

figure;

plot(t, omega, 'LineWidth', 2);

grid on;

xlabel('Tempo [s]');
ylabel('\omega [rad/s]');

title('Velocidade angular do veiculo');

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

fprintf('=============================================\n');

