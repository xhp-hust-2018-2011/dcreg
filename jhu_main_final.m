RootDir = './'
DsetName = 'JHU_resize';
DsetPath = fullfile(RootDir,'jhu_crowd_v2.0');
SplitList = {'train','test'};
SaveRootDir = 'Processed_Dataset';
SaveRootDir = fullfile(SaveRootDir,DsetName);
RESIZE = true;
MAXSIZE = 2048;
for spi = 1:length(SplitList)
    img_save_dir =  fullfile( SaveRootDir, SplitList{spi}, 'images');
    gtdens_save_dir = fullfile( SaveRootDir, SplitList{spi}, 'gtdens');
    if ~exist(img_save_dir)
        mkdir(img_save_dir)
    end
    if ~exist(gtdens_save_dir)
        mkdir(gtdens_save_dir)
    end
    tmp_dset_root = fullfile(DsetPath,SplitList{spi});
    % load full filelist into filelist
    % This is for gt.Txtformate imgid gt place unknownindex1 unknownindex2
    main_txt_path = fullfile(tmp_dset_root,'image_labels.txt');
    [filelist, num_gt, location, a, b] = textread(main_txt_path, '%s%n%s%n%n', 'delimiter', ',');
    % img can be read by the id in idx
    num_imgs = length(filelist);

    % ProcessImage
    for i = 1:num_imgs
        % read imgs
        img_dir = fullfile(tmp_dset_root,'images',strcat(filelist{i},'.jpg'));
        I = imread(img_dir);
        if length(size(I))==2
            I = repmat(I,1,1,3);
        end
        % read gt
        gt_dir = fullfile(tmp_dset_root,'gt',strcat(filelist{i},'.txt'));
        [x,y,w,h,a5,a6] = textread(gt_dir, '%n%n%n%n%n%n');
        % check if all points are within border
        x_false = find( x> size(I, 2) );
        y_false = find( y> size(I, 1) );
        xy_false = union( x_false,y_false );
        if length(xy_false)>0
          x(xy_false,:)=[];
          y(xy_false,:)=[];
        end
        x_false = find( x<0 );
        y_false = find( y<0 );
        xy_false = union( x_false,y_false );
        if length(xy_false)>0
          x(xy_false,:)=[];
          y(xy_false,:)=[];
        end


        % check if resize image
        [imh,imw,~] = size(I);
        if max(imh,imw)>MAXSIZE && RESIZE
            r = MAXSIZE/max(imh,imw);
            new_imh = ceil(imh*r);
            new_imw = ceil(imw*r);
            I = imresize(I,[new_imh,new_imw],'bicubic','Antialiasing',true);
            x = ceil(x*r);
            y = ceil(y*r);
            w = ceil(w*r);
            h = ceil(h*r);

        end

        % ensure the minumum is 1
        x = max(x,1);
        y = max(y,1);

        anno_num = length(x);
        
        pnum = length(x);
        density_map = get_2D_gaussian_densitymap(I,x,y,w,h);
        
        % save file
        img_save_path = fullfile(img_save_dir,strcat(filelist{i},'.jpg'));
        imwrite(I ,img_save_path );
        gtdens_save_path = fullfile( gtdens_save_dir,strcat(filelist{i},'.mat'));
        dot_map = zeros(size(I, 1), size(I, 2), 'single');
        if ~isempty(y)
            dot_map(sub2ind(size(dot_map), y, x)) = 1;
        end
        save(gtdens_save_path,'dot_map','density_map','anno_num','-v6')
        fprintf('%d/%d-%s.jpg has been processed!\n',i,num_imgs,filelist{i})
    end
 
end
