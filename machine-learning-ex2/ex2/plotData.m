function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%









% =========================================================================
m =1;
n =1;

for i = 1:length(y)
  if y(i)==0
    x1(m,1) = X(i,1);
    x1(m,2) = X(i,2);
    m = m+1;
  elseif y(i) == 1
    x2(m,1) = X(i,1);
    x2(m,2) = X(i,2);
    n = n+1;
  end
end 
plot(x2(:,1),x2(:,2),'+'); hold on;

plot(x1(:,1),x1(:,2),'o'); hold on;

hold off;

end
