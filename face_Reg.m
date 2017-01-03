function face_Reg()
test_indx = [1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5 6 6 6 6 7 7 7 7 8 8 8 8 9 9 9 9 10 10 10 10 11 11 11 11 12 12 12 12 13 13 13 13 14 14 14 14 15 15 15 15];
    test_dir = dir('/Users/vhineshravi/Desktop/cv/faceReg/test');
    d=0;
    acc=0;
    % Reading and testing all images from test directory
    for i = 1:size(test_dir,1)
        if not(strcmp(test_dir(i).name,'.')|strcmp(test_dir(i).name,'..')|strcmp(test_dir(i).name,'.DS_Store'))
            im_name = strcat('/Users/vhineshravi/Desktop/cv/faceReg/test/',test_dir(i).name);
            d=d+1;
            test_im = imread(im_name);
            face_db = trainProcess();
            [face_space,avg,eigen_face] = eigenFaceModel(face_db);
            proj_Space = testProcess(test_im,avg,eigen_face,face_space);
            % Finding eucledian distance between test image and the images
            % in training set.
            eucledian_dist = [];
            for i = 1 : size(eigen_face,2)
                temp = ( norm(proj_Space  - face_space(:,i) ) );
                eucledian_dist = [eucledian_dist temp];
            end
            % Finding index of training image with minimum distance 
            [~, id]=min(eucledian_dist);
            id=ceil(id/7);
            
            if (test_indx(d)==id)
                acc=acc+1;
            end
                
        end
    end
    fprintf ( 'accuracy  %f', (double(acc)/double(d)) );
    %test_im = imread('/Users/vhineshravi/Desktop/cv/faceReg/test/subject09.happy.gif');
    
end
