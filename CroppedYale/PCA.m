function [COEFF, SCORE,latent] = PCA(X)
    %my own PCA function
    x_size = size(X);
    x_len = x_size(1);
    X = X - ones(x_len,1)*mean(X,1);
    [U, S, V] = svd(X);
    COEFF = V;
    SCORE = U*S;
    %move all latent into a vector
    latent = [];
    L = S' * S;
    x_size = size(L);
    x_length = x_size(1);
    for i=1:x_length
        latent = [latent;L(i,i)];
    end
end

