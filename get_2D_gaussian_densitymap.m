function im_density = get_2D_gaussian_densitymap(im,X,Y,W,H,varargin)
% %!already prob map
% opt.adaptive_gauss = true;
% opt.sigma = 8;
% opt.f_sz = 41;
% [opt,varargin] = vl_argparse(opt,varargin);
opt.SZ_X = max(W,4);
opt.SZ_Y = max(H,4);
opt.SIGMA_X = opt.SZ_X;%/2;
opt.SIGMA_Y = opt.SZ_Y;%/2;

[h,w,~]=size(im);
im_density = zeros(h,w,'single');

if isempty(X)
  return;
end

for pi = 1:length(X)
  
  H = get_2d_Gaussian(opt.SZ_X(pi),opt.SZ_Y(pi),opt.SIGMA_X(pi),opt.SIGMA_Y(pi));
  x = min( w,max(1,X(pi)) );
  y = min( h,max(1,Y(pi)) );
  x1 = x - floor(opt.SZ_X(pi)/2); y1 = y - floor(opt.SZ_Y(pi)/2);
  x2 = x1 + opt.SZ_X(pi)-1; y2 = y1 + opt.SZ_Y(pi)-1;
  dfx1 = 0; dfy1 = 0; dfx2 = 0; dfy2 = 0;
  change_H = true;
  if(x1 < 1)
    dfx1 = abs(x1)+1;
    x1 = 1;
    change_H = true;
  end
  if(y1 < 1)
    dfy1 = abs(y1)+1;
    y1 = 1;
    change_H = true;
  end
  if(x2 > w)
    dfx2 = x2 - w;
    x2 = w;
    change_H = true;
  end
  if(y2 > h)
    dfy2 = y2 - h;
    y2 = h;
    change_H = true;
  end
  x1h = 1+dfx1; y1h = 1+dfy1; x2h = opt.SZ_X(pi) - dfx2; y2h = opt.SZ_Y(pi) - dfy2;
  if (change_H == true)
    H = H(y1h:y2h,x1h:x2h);
  end
  
  im_density(y1:y2,x1:x2) = im_density(y1:y2,x1:x2)+H ;
end

end


function H = get_2d_Gaussian(szx,szy,sigmax,sigmay)
x=1:1:szx;
y=1:1:szy;
[X,Y]=meshgrid(x,y);
mu = [(1+szx)/2.0,(1+szy)/2.0];
sigma = [sigmax,0;
        0,sigmay];
% mu
% sigma
H = mvnpdf([X(:),Y(:)],mu,sigma);

H = reshape(H,size(X));
% norm to 1
H = H/sum(H(:));

end

