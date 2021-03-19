%% Amath482 HW5 Code

% Clean workspace
clear all; close all; clc

%% Set up

% Load Video 1
vid1 = VideoReader("ski_drop_low.mp4"); % video file name can be replaced
vidFrames = read(vid1);
[height, width, RGB, numFrames] = size(vidFrames);

% Watch Movie
for i=1:numFrames
    X = vidFrames(:,:,:,i);
    % imshow(X); drawnow
end

% Crop out edges and convert to grayscale
numRows = 500-49;
numCols = 600-299;
gray_vid = zeros(numRows,numCols,numFrames);

for j=1:numFrames
    gimage = rgb2gray(vidFrames(50:500,300:600,:,j));
    gray_vid(:,:,j) = abs(255-gimage);
    % imshow(abs(255-gimage)); drawnow
end

%% Set up DMD

X = reshape(gray_vid, numRows*numCols, numFrames);

height = numRows;
width = numCols;

X1 = X(:,1:end-1);
X2 = X(:,2:end);
r = 2;
dt = 1/ vid1.Framerate;

% DMD
[U, S, V] = svd(X1,'econ');
r = min(r, size(U,2));
U_r = U(:, 1:r); % truncate to rank-r
S_r = S(1:r, 1:r);
V_r = V(:, 1:r);
Atilde = U_r' * X2 * V_r / S_r; % low-rank dynamics
[W_r , D] = eig(Atilde);
Phi = X2 * V_r / S_r * W_r; % DMD modes
lambda = diag(D); % discrete -time eigenvalues
omega = log(lambda)/dt; % continuous-time eigenvalues

% Compute DMD mode amplitudes
x1 =X1(:, 1);
b = Phi\x1;

% DMD reconstruction
mm1 = size(X1, 2); % mm1 = m - 1
time_dynamics = zeros(r, mm1);
t = (0:mm1 - 1)*dt; % time vector
for iter = 1:mm1
    time_dynamics(:, iter) = (b.*exp(omega*t(iter)));
end
Xdmd = Phi * time_dynamics;

%% Separate background and foreground
og = vidFrames(50:500,300:600,:,:);
bg = uint8(Xdmd);
fg = uint8(X(:,1:numFrames-1) - Xdmd);

% reshaped videos
for i=1:numFrames-1
    bg_frame(:,:,:,i) = reshape(bg(:,i),height,width);
end

for j=1:numFrames-1
    fg_frame(:,:,:,j) = reshape(fg(:,j),height,width);
end

%% plot SVD modes
sig = diag(S)/sum(diag(S));
figure(1);
subplot(2,1,1), plot(sig,'ko','Linewidth',[1.1])
title('Singular Values: Ski Drop')
ylabel('Energy, %')
subplot(2,1,2), semilogy(sig,'ko','Linewidth',[1.1])
ylabel('log(energy)')
xlabel('Singular Values')

%% plot original, background, and foreground at frames 2, 20, and 30
figure(2);
subplot(3,3,1), imagesc(rgb2gray(og(:,:,:,2))), colormap(gray), axis off
subplot(3,3,2), imagesc(rgb2gray(og(:,:,:,20))), colormap(gray), axis off
title('Background Subtraction of Video 1')
subplot(3,3,3), imagesc(rgb2gray(og(:,:,:,30))), colormap(gray), axis off

subplot(3,3,4), imagesc(bg_frame(:,:,:,2)), colormap(gray), axis off
subplot(3,3,5), imagesc(bg_frame(:,:,:,20)), colormap(gray), axis off
subplot(3,3,6), imagesc(bg_frame(:,:,:,30)), colormap(gray), axis off

subplot(3,3,7), imagesc(fg_frame(:,:,:,2)), colormap(gray), axis off
subplot(3,3,8), imagesc(fg_frame(:,:,:,20)), colormap(gray), axis off
subplot(3,3,9), imagesc(fg_frame(:,:,:,30)), colormap(gray), axis off