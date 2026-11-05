%% ============================================================
%  GROUND TRUTH - ROBÔ DIFERENCIAL 2WD
%
%  Roda direita  = 30 deg/s
%  Roda esquerda = 15 deg/s
%
%  O ground truth é calculado ANALITICAMENTE.
%
%  Estado:
%       X = [x y theta]'
%
%  Unidade:
%       x,y     -> m
%       theta   -> rad
% =============================================================

clear;
close all;
clc;

%% -------------------------------------------------------------
% 1. PARÂMETROS DO ROBÔ
% -------------------------------------------------------------

r = 0.03;              % Raio da roda [m]
L = 0.15;              % Distância entre rodas [m]

Ts = 0.1;              % Período de amostragem [s]
fs = 1/Ts;             % Frequência de amostragem [Hz]

Tsim = 120;             % Tempo total de simulação [s]

%% -------------------------------------------------------------
% 2. VELOCIDADES ANGULARES DAS RODAS
% -------------------------------------------------------------

omega_R_deg = 30;      % Roda direita [deg/s]
omega_L_deg = 15;      % Roda esquerda [deg/s]

% Conversão para rad/s
omega_R = deg2rad(omega_R_deg);
omega_L = deg2rad(omega_L_deg);

%% -------------------------------------------------------------
% 3. VELOCIDADES LINEARES DAS RODAS
% -------------------------------------------------------------

v_R = r * omega_R;
v_L = r * omega_L;

%% -------------------------------------------------------------
% 4. VELOCIDADE DO ROBÔ
% -------------------------------------------------------------

% Velocidade linear do centro do robô
v = (v_R + v_L)/2;

% Velocidade angular do robô
omega = (v_R - v_L)/L;

% Raio instantâneo da trajetória
R = v/omega;

%% -------------------------------------------------------------
% 5. INFORMAÇÕES DO SISTEMA
% -------------------------------------------------------------

fprintf('\n');
fprintf('===============================================\n');
fprintf('        PARAMETROS DO ROBO 2WD\n');
fprintf('===============================================\n');

fprintf('Raio da roda             = %.4f m\n', r);
fprintf('Distancia entre rodas    = %.4f m\n', L);

fprintf('\n');

fprintf('Velocidade roda direita  = %.4f deg/s\n', ...
    omega_R_deg);

fprintf('Velocidade roda esquerda = %.4f deg/s\n', ...
    omega_L_deg);

fprintf('\n');

fprintf('Velocidade roda direita  = %.6f m/s\n', ...
    v_R);

fprintf('Velocidade roda esquerda = %.6f m/s\n', ...
    v_L);

fprintf('\n');

fprintf('Velocidade linear robo   = %.6f m/s\n', ...
    v);

fprintf('Velocidade angular robo  = %.6f rad/s\n', ...
    omega);

fprintf('Velocidade angular robo  = %.4f deg/s\n', ...
    rad2deg(omega));

fprintf('\n');

fprintf('Raio da trajetoria       = %.6f m\n', ...
    R);

fprintf('===============================================\n');

%% -------------------------------------------------------------
% 6. VETOR DE TEMPO
% -------------------------------------------------------------

t = 0:Ts:Tsim;

N = length(t);

%% -------------------------------------------------------------
% 7. GROUND TRUTH ANALÍTICO
% -------------------------------------------------------------

% Condição inicial:
%
% x(0)     = 0
% y(0)     = 0
% theta(0) = 0
%
% A trajetória é um arco de circunferência.

x_gt = R * sin(omega*t);

y_gt = R * (1 - cos(omega*t));

theta_gt = omega*t;

%% -------------------------------------------------------------
% 8. ESTADO FINAL
% -------------------------------------------------------------

fprintf('\n');
fprintf('===============================================\n');
fprintf('             ESTADO FINAL\n');
fprintf('===============================================\n');

fprintf('Tempo       = %.2f s\n', t(end));

fprintf('x           = %.6f m\n', x_gt(end));
fprintf('y           = %.6f m\n', y_gt(end));

fprintf('theta       = %.6f rad\n', theta_gt(end));
fprintf('theta       = %.6f deg\n', rad2deg(theta_gt(end)));

fprintf('\n');

fprintf('Numero de amostras = %d\n', N);

fprintf('===============================================\n');

%% -------------------------------------------------------------
% 9. TRAJETÓRIA DO ROBÔ
% -------------------------------------------------------------

figure('Name','Ground Truth - Trajetoria');

plot(x_gt, y_gt, 'LineWidth', 2);

hold on;

% Posição inicial
plot(x_gt(1), y_gt(1), 'o', ...
    'MarkerSize', 8, ...
    'LineWidth', 2);

% Posição final
plot(x_gt(end), y_gt(end), 's', ...
    'MarkerSize', 8, ...
    'LineWidth', 2);

grid on;
axis equal;

xlabel('x [m]');
ylabel('y [m]');

title('Ground Truth - Trajetória do Robô');

legend('Trajetória', ...
       'Inicial', ...
       'Final', ...
       'Location','best');

%% -------------------------------------------------------------
% 10. POSIÇÃO AO LONGO DO TEMPO
% -------------------------------------------------------------

figure('Name','Ground Truth - Posicao');

plot(t, x_gt, 'LineWidth', 1.8);

hold on;

plot(t, y_gt, 'LineWidth', 1.8);

grid on;

xlabel('Tempo [s]');
ylabel('Posição [m]');

title('Ground Truth - Posição');

legend('x(t)', 'y(t)', ...
       'Location','best');

%% -------------------------------------------------------------
% 11. ORIENTAÇÃO
% -------------------------------------------------------------

figure('Name','Ground Truth - Orientacao');

plot(t, rad2deg(theta_gt), 'LineWidth', 1.8);

grid on;

xlabel('Tempo [s]');
ylabel('\theta [graus]');

title('Ground Truth - Orientação');

%% -------------------------------------------------------------
% 12. VELOCIDADE ANGULAR
% -------------------------------------------------------------

figure('Name','Ground Truth - Velocidade Angular');

plot(t, rad2deg(omega)*ones(size(t)), ...
    'LineWidth', 1.8);

grid on;

xlabel('Tempo [s]');
ylabel('\omega [graus/s]');

title('Velocidade Angular do Robô');

%% -------------------------------------------------------------
% 13. VERIFICAÇÃO DA EQUAÇÃO DA TRAJETÓRIA
% -------------------------------------------------------------

% Equação do círculo:
%
% (x)^2 + (y-R)^2 = R^2

circle_error = ...
    sqrt(x_gt.^2 + (y_gt - R).^2) - R;

fprintf('\n');
fprintf('Erro máximo na equação do círculo:\n');
fprintf('%.12e m\n', max(abs(circle_error)));

%% -------------------------------------------------------------
% 14. SALVAR GROUND TRUTH
% -------------------------------------------------------------

ground_truth.t = t;
ground_truth.x = x_gt;
ground_truth.y = y_gt;
ground_truth.theta = theta_gt;

ground_truth.r = r;
ground_truth.L = L;

ground_truth.v_R = v_R;
ground_truth.v_L = v_L;

ground_truth.v = v;
ground_truth.omega = omega;
ground_truth.R = R;

save('ground_truth_2WD.mat', 'ground_truth');

fprintf('\nGround truth salvo em:\n');
fprintf('ground_truth_2WD.mat\n');