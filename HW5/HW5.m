%% Amath482 HW5 Code

% Clean workspace
clear all; close all; clc

%% Set up

% Load Video
vid1 = VideoReader("ski_drop_low.mp4");
vid1_frames = read(vid1);
vid1_dt = 1/ vid1.Framerate ;
vid1_t = 0: vid1_dt: vid1.Duration ;
[height, width, RGB, numFrames] = size(vid1);

% Watch Movie
for i=1:numFrames1
    X = vid1(:,:,:,i);
    % imshow(X); drawnow
end

% Crop out edges and convert to grayscale
numRows = 500-49;
numCols = 600-299;
gray_vid = zeros(numRows,numCols,numFrames);

for j=1:numFrames
    gimage=rgb2gray(vid1(50:500,300:600,:,j));
    gray_vid(:,:,j) = abs(255-gimage);
    imshow(abs(255-gimage)); drawnow
end

% DMD
X1 = X(:,1:end-1);
X2 = X(:,2:end);

[U,S,V] = svd(X1,'econ');
r = 2;

U_r = U(:, 1:r);
S_r = S(1:r, 1:r);
V_r = V(:, 1:r);
A_tilde = U_r' * X2 * V_r / S_r;
[W_r,D] = eig(A_tilde);
Phi = X2 * V_r / S_r * W_r;

lambda = diag(D);
omega = log(lambda)/v1_dt;
x1 =X1(:, 1);
b = Phi \ x1;

mm1 = size(X1,2);
time_dynamics = zeros(r,mm1);
t = (0:mm1-1)*v1_dt;
for k = 1:mm1
    time_dynamics(:,k) = (b.*exp(omega*t(k)));    
end
Xdmd = Phi * time_dynamics;

% 