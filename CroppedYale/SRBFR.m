function [ACCURACY] = SRBFR(numTrainSamples, filePath)
    training_imgs = {};         %record the file name of the training images
    training_matrix = [];
    %get the sub directory's names
    dirs = dir(filePath);
    size_a = size(dirs);
    a = size_a(1);
    sub_dirs = {};      %record the sub_dirs
    for i = 1:a
        %get "yaleB{x}"directory name
        if regexp(dirs(i).name,'^yale') ~= 0
            sub_dir = fullfile(filePath,dirs(i).name);
            sub_dirs = [sub_dirs , sub_dir];
        end
    end

    size_b = size(sub_dirs);
    sub_dir_nums = size_b(2);
    %get the training data
    for i = 1:sub_dir_nums
        files_raw = dir(char(sub_dirs(i)));
        files_raw_size = size(files_raw);
        files_raw_num = files_raw_size(1);
        %remove invalid files
        files_name = {};    %store the valid file path & name
        for j=1:files_raw_num
            %get files with .pgm and without Ambient
            if (regexp(files_raw(j).name,'^yale') ~= 0) & (regexp(files_raw(j).name,'.pgm') ~= 0) & (isempty(regexp(files_raw(j).name,'Ambient','once')) )
                file_name = fullfile(char(sub_dirs(i)),files_raw(j).name);
                files_name = [files_name,file_name];
            end
        end
        %get the numbers of valid files
        files_size = size(files_name);
        files_num = files_size(2);
        %get training datas
        for k=1:numTrainSamples
            while(1)
                rand_num = fix(rand() * files_num) + 1;
                img_path = char(files_name(rand_num));
                %to aviod same image
                if ~checkExistTrainingImg(img_path,training_imgs)
                    training_imgs = [training_imgs, img_path];
                    img_v = readImg(img_path);
                    training_matrix = [training_matrix img_v];
                    break;
                end
            end
        end
    end
    
    training_matrix_h = training_matrix';
    [COEFF,SCORE,latent] = PCA(training_matrix_h);
    %cut the least principal cmponment of the matrix
    total_latent = sum(latent);
    temp_total = 0;
    SCORE_new = [];
    m = 1;
    while(1)
        if (temp_total > 0.95 * total_latent)
            break;
        end
        SCORE_new = [SCORE_new SCORE(:,m)];
        temp_total = temp_total + latent(m);
        m = m + 1;
    end
    
    %start testing
    correct = 0;
    total = 0;
    for i = 1:sub_dir_nums
        ground_truth = i;
        files_raw = dir(char(sub_dirs(i)));
        files_raw_size = size(files_raw);
        files_raw_num = files_raw_size(1);
        %remove invalid files
        files_name = {};    %store the valid file path & name
        for j=1:files_raw_num
            %get files with .pgm and without Ambient
            if (regexp(files_raw(j).name,'^yale') ~= 0) & (regexp(files_raw(j).name,'.pgm') ~= 0) & (isempty(regexp(files_raw(j).name,'Ambient','once')) )
                file_name = fullfile(char(sub_dirs(i)),files_raw(j).name);
                files_name = [files_name,file_name];
            end
        end
        %get the numbers of valid files
        files_size = size(files_name);
        files_num = files_size(2);
        %the actual testing part
        for k=1:files_num
            img_path = char(files_name(k));
            if ~checkExistTrainingImg(img_path,training_imgs)
                total = total + 1;
                %if the image is not a trainging image, test it
                img_test = readImg(img_path);
                label = getFace(img_test,training_matrix,COEFF,SCORE_new,latent,numTrainSamples,m-1);
                if label == ground_truth
                    correct = correct + 1;
                end
            end
        end
        %uncomment if you want to update and print accuracy out every time it complete a
        %subdirectory computing
        %ACCURACY = correct/total
    end
    
    ACCURACY = correct/total;
    
end

%the score here is the new score only keeping 95% of the principle componments
function label = getFace(test_vector_ori,training_matrix,COEFF,SCORE,latent,numTrainSamples,vector_len)
    v = 0.1;
    %basis change
    mean_training = mean(training_matrix',1);
    test_vector = test_vector_ori' - mean_training;
    test_vector = test_vector * COEFF;
    %cut the least principal cmponment of the vector
    test_vector_new =[];
    for k = 1:vector_len
        test_vector_new = [test_vector_new test_vector(1,k)];
    end
    x_size = size(SCORE);
    x_length = x_size(1);
    x = feature_sign(SCORE',test_vector_new',0.005,zeros(x_length,1));
    min_err = -1;
    min_img = -1;
    for index = 1:x_length
        if x(index) > v
            err = norm(test_vector_new' - SCORE(index)');
            if (err < min_err) | (min_err == -1)
                min_err = err;
                min_img = index;
            end
        end
    end
    label = fix((min_img-1)/numTrainSamples) + 1;
end

function exist_flag =  checkExistTrainingImg(target,string_array)
    array_size = size(string_array);
    ele_num = array_size(2);
    exist_flag = 0;
    for i=1:ele_num
        if regexp(char(string_array(i)),target)
            exist_flag = 1;
            break;
        end
    end
    
end

%read in an imagine from the given path
function img_v = readImg(img_path)
    img_m = imresize(imread(img_path), 0.5);
    %img_m = imread(img_path);
    img_m = single(img_m);   %the image matrix
    img_v = reshape(img_m,[],1);
    img_v = img_v / norm(img_v);    %normalize it
end