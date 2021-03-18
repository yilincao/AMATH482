% Clean workspace
clear all; close all; clc

load subdata.mat % Imports the data as the 262144x49 (space by time) matrix called subdata

%% Set up

L = 10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); x = x2(1:n); y = x; z = x;
k = (2*pi/(2*L))*[0:(n/2-1) -n/2:-1]; ks = fftshift(k);

[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

%% Step 1.

% Average of the spectrum
ave = zeros(n,n,n);
for j=1:49
    ave = ave + fftn(reshape(subdata(:,j),n,n,n));
end
ave = abs(fftshift(ave))/49;
maxAve = max(abs(ave),[],'all');

% Plot averaged signal
figure(1)
isosurface(Kx,Ky,Kz,ave./max(ave(:)),0.6)
axis([-10 10 -10 10 -10 10]), grid on, drawnow
xlabel("Kx"), ylabel("Ky"), zlabel("Kz")
title("Signals in frequency domain");

% Determine the center frequency
[a,b,c] = ind2sub([n,n,n],find(abs(ave) == maxAve));
x_cf = ks(b);
y_cf = ks(a);
z_cf = ks(c);

%% Step 2.

% Define the Gaussian Filter
tau = 0.2;
filter = exp(-tau * ((Kx - x_cf).^2 + (Ky - y_cf).^2 + (Kz - z_cf).^2));

% Determine the path of the submarine
path = zeros(49,3);
for i = 1:49
    un(:,:,:) = reshape(subdata(:,i),n,n,n);
    utn = fftshift(fftn(un));
    unft = filter.*utn; % Apply the filter to the signal in frequency space
    unf = ifftn(unft);
    maxUnf = max(abs(unf),[],'all');
    [pathX,pathY,pathZ] = ind2sub([n,n,n], find(abs(unf)==maxUnf));
    path(i,1) = X(pathX,pathY,pathZ);
    path(i,2) = Y(pathX,pathY,pathZ);
    path(i,3) = Z(pathX,pathY,pathZ);
end

% Plot the path of the submarine
figure(2)
plot3(path(:,1),path(:,2),path(:,3),'b-o','LineWidth',1.5);
axis([-10 10 -10 10 -10 10]), grid on, drawnow
title('The Path of the Submarine');
xlabel("x"), ylabel("y"), zlabel("z")
hold on;
plot3(path(49,1),path(49,2),path(49,3),'r*','MarkerSize',15);
hold off;

%% Step 3.

% Location to send P-8 Poseidon subtracking aircraft
loc = path(end,:);
isosurface(X,Y,Z,abs(unf)./max(abs(unf(:))),0.5)
axis([-10 10 -10 10 -10 10]), grid on, drawnow
xlabel("X"), ylabel("Y"), zlabel("Z")
title("Final position of the submarine");