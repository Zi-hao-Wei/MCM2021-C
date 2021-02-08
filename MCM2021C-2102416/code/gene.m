% This code is just an example. It can not run!!!!

absbs=13;% need adjust 
B_su1=[zeros(n,1);0.00001];
%B_sb1 means the VLB of the Xi
B_b2=[ones(n,1)*2100;0.1];
%B_ub2 means the VUB of the Xi
B=[B_s,B_b2];
% consint of Xi
init=inliadaszega(n,B,'fitness');
[x enp,bp,tre]=ga(B,'fess',[],inop,[1e-6 1 1],
	'maxGTerm',100,'noomSelect',...
    [0.08],['arover'],[2],'nnUMutation',[2 10000 3]); 
%3000 erations
xx=so(x(1:14));
% sis selection sorts as ascending order
x=[x,ad(15)];
% the last one is the on value