function [proj_Space] = testProcess(test_im,avg,eigen_face,face_space)
    % Converting image to double
    test_im = im2double(test_im);
    % Face detuction from input image
    FDetect = vision.CascadeObjectDetector;
    BB = step(FDetect,test_im);
    % Cropping the face from the input image
    test_im= imcrop(test_im,BB(1,:));
    % Resisizing the cropped face
    test_im = imresize(test_im, [141 141]);
    % Exposure enhancement
    test_im = ((test_im).^(1/2));
    % Smoothing using gauss filter
    fltr        = fspecial( 'gauss', 5, 2 );
    test_im     = imfilter( test_im, fltr, 'same', 'repl' );
    %test_im = sqrt(test_im);
    %test_im = adapthisteq(test_im);
    [row,col] = size(test_im);
    % Reshaping 2D images into 1D vectors
    vec_img = reshape(test_im',row*col,1); 
    % Subtracting mean face from the test image
    vec_img = double(vec_img)-avg;
    % Projecting into eigen space
    proj_Space = eigen_face'* vec_img;

   
end