%% Amath482 HW2 Code

% Clean workspace
clear all; close all; clc

%% Part 1. GNR Clip
figure(1)
[y, Fs] = audioread('GNR.m4a');
tr_gnr = length(y)/Fs; % record time in seconds
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Sweet Child O'' Mine');
% p8 = audioplayer(y,Fs); playblocking(p8);

% Set up
y = y.'; n = length(y); L = tr_gnr;
t1 = linspace(0,L,n+1); t = t1(1:n);
k = (1/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);

% Apply Gabor Transformation
a = 500;
tau = 0:0.1:L;
Sgt_spec = [];
gnr_guitar = [];
for j = 1:length(tau)
    g = exp(-a*(t-tau(j)).^2);
    Sg = g.*y;
    Sgt = fft(Sg);
    Sgt_spec(:,j) = fftshift(abs(Sgt));
    [M,I] = max(Sgt);
    gnr_guitar = [gnr_guitar; abs(k(I))];
end

% Plot the Spectrogram of GNR clip
figure(2)
pcolor(tau,ks,Sgt_spec)
shading interp
colormap(hot)
colorbar
set(gca,'Fontsize',12,'ylim',[200 800]);
xlabel('Time [sec]'); ylabel('Frequency [Hz]');
title('Spectrogram of Sweet Child O'' Mine (guitar)');

% Plot the music score for the guitar in GNR clip
figure(3)
plot(tau, gnr_guitar, 'ko', 'MarkerFaceColor', 'r')
xlabel('Time (sec)'), ylabel('Frequency (Hz)')
yticks([277.18, 311.13, 369.99, 415.30, 554.37, 698.46, 739.99])
yticklabels({'C4#', 'D4#', 'F4#', 'G4#', 'C5#', 'F5', 'F5#'})
set(gca,'Fontsize',12,'ylim',[200 800])
title('Music score for the guitar in GNR clip')

%% Part 1. Floyd Clip
figure(4)
[y, Fs] = audioread('Floyd.m4a');
tr_floyd = length(y)/Fs; % record time in seconds
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Comfortably Numb');
% p8 = audioplayer(y,Fs); playblocking(p8);

% Set up
L = 15; n = L*Fs; y = y(1:n).';
t1 = linspace(0,L,n+1); t = t1(1:n);
k = (1/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);

% Apply Gabor Transformation
a = 500;
tau = 0:0.1:L;
Sgt_spec = [];
floyd_bass = [];
for j = 1:length(tau)
    g = exp(-a*(t-tau(j)).^2);
    Sg = g.*y;
    Sgt = fft(Sg);
    Sgt_spec(:,j) = fftshift(abs(Sgt));
    [M,I] = max(Sgt);
    floyd_bass = [floyd_bass; abs(k(I))];
end

% Plot the Spectrogram of Floyd Clip
figure(5)
pcolor(tau,ks,Sgt_spec)
shading interp
colormap(hot)
colorbar
set(gca,'Fontsize',12,'ylim',[60 160]);
xlabel('time (t)'); ylabel('frequency (k)');
title('Spectrogram of Comfortably Numb (bass)');

% Plot the music score for the bass in GNR clip
figure(6)
plot(tau, floyd_bass, 'ko', 'MarkerFaceColor', 'r')
xlabel('Time (sec)'), ylabel('Frequency (Hz)')
yticks([82.407, 92.499, 97.999, 110.00, 123.47])
yticklabels({'E2', 'F2#', 'G2', 'A2', 'B2'})
set(gca,'Fontsize',12,'ylim',[60 160])
title('Music score for the bass in Floyd clip')

%% Part 2. Isolate the bass in Comfortably Numb
[y, Fs] = audioread('Floyd.m4a');
tr_floyd = length(y)/Fs; % record time in seconds
y = y(1:end-1);
yt = fft(y);

y = y.'; n = length(y); L = tr_floyd;
k = (1/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);

% Shannon filter
[Ny,Nx] = size(y);
wy = 10000;
sfilter = ones(size(y));
sfilter(wy+1:Nx-wy+1) = zeros(1,Nx-2*wy+1);

ytf = yt.*sfilter;
yf = ifft2(ytf);

figure(7)
subplot(2,1,1)
plot(ks,abs(fftshift(yt)),'r')
hold on
plot(ks, fftshift(sfilter),'k','Linewidth',2)

subplot(2,1,2)
plot(ks,abs(yf),'Linewidth',2);