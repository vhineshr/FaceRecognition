function face_db = trainProcess()
    
    training = dir('/Users/vhineshravi/Desktop/cv/faceReg/training');
    face_db = [];
    d=0;
    for i = 1:size(training,1)
        
        if not(strcmp(training(i).name,'.')|strcmp(training(i).name,'..')|strcmp(training(i).name,'.DS_Store'))
            
            im_name = strcat('/Users/vhineshravi/Desktop/cv/faceReg/training/',training(i).name);
            % Reading input image
            image = imread(im_name);
            % Converting image to double
            image = im2double(image);
            % Face detuction from input image
            FDetect = vision.CascadeObjectDetector;
            BB = step(FDetect,image);
            % Cropping the face from the input image
            image= imcrop(image,BB(1,:));
            % Resisizing the cropped face
            image = imresize(image, [141 141]);
            % Exposure enhancement
            image = ((image).^(1/2));
            % Smoothing using gauss filter
            fltr        = fspecial( 'gauss', 5, 2 );
            image          = imfilter( image, fltr, 'same', 'repl' );
           
            %image = adapthisteq(image);
            
            [row, col] = size(image);
            % Reshaping 2D images into 1D vectors
            image_vector = reshape(image',row*col,1);   
            face_db = [face_db image_vector]; 
        end
           
    end
end
