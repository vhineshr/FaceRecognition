function [face_space,avg,eigen_face] = eigenFaceModel(face_db)
    
    % calculating mean face vector
    avg = mean(face_db,2); 
    
    %l=reshape(avg(:,1), [141 141])';
    %imshow(l);
    %imwrite(l,'face_mean.jpg')
    
    % Normalizing training images by subtracting mean face.
    centered_face = [];  
    for i = 1 : size(face_db,2)
        temp = double(face_db(:,i)) - avg;
        centered_face = [centered_face temp]; 
    end
    
    % Finding Eigen values and eigen vectors from covariance matrix
    [V,D]= eig(centered_face'*centered_face);
    i_d=[];
    e_v=[];
    
    for j=1:size(V,2)
        % only eigen vectors corresponding to eigen values greter than 1
        % are selected.
        if(D(j,j)>1)
                
            e_v=[e_v V(:,j)];
            i_d=[i_d D(j,j)];
            
        end
    end
    %getting the eigen values and sorting in decresing order
    x=diag(D);
    [xv,order]=sort(x,'descend');
    ef_im=V(:,order);
    e=[];
    
    eigen_face = centered_face * e_v;
    
    
    %size(xi)
    %figure;
    
    
    %size(ef_im)
    %size(eigen_face)
    
    % plotting no of principal components vs cumulative variance.
    %normalised_evalues = xv / sum(x);
    %figure; 
    %plot(cumsum(normalised_evalues));
    %xlabel('No of Principal Components'), ylabel('Variance');
    %xlim([1 50]), ylim([0 1]), grid on;
    
    %Projection of normalized images into eigen space
    face_space = [];
    for i = 1 : size(eigen_face,2)
        temp = eigen_face'*centered_face(:,i); 
       
        face_space = [face_space temp]; 
    end

end
