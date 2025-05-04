运行命令
python main.py --source DATA --slide_ext .svs --step_size 256 --patch_size 256 --batch_size 256  --device cuda

参数说明
--source 文件夹位置（不能包含中文）
--slide_ext 文件后缀
 --step_size 滑动步长
--patch_size 滑动窗口长度
--batch_size  batch size
--device  在GPU运行“cuda” 在CPU运行“cpu”

说明：
1、切片文件放到--source文件夹里面，文件路径不能包含中文；
2、处理过程中会产生缓存文件放在当前目录下，处理完成后会删除；
3、程序生成的结果（png文件）默认放在当前目录下“result_data”文件夹中，也可以通过--result_dir命令自行指定存放位置；
4、特征提取网络预训练权重下载链接：https://drive.google.com/file/d/1_XqBHTxEk6fQhUi9N5oiGSegcjK5UzYP/view?usp=drive_link，下载后放到根目录下。