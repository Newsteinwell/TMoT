# 第二轮训练源版本

|实验|summary中的源标识|不可变Git提交|
|---|---|---|
|diagnosis_v2|d8735b5|d8735b5|
|transport_v3|ae4055a|ae4055a|
|followups_v4|8c73062|8c73062|
|locked_test_v5|locked_test_v5|ceb8556|

locked_test_v5运行环境使用了配置别名作为TMOT_SOURCE_COMMIT，故summary的git_commit字段不是SHA。其实际代码提交为ceb8556；保留原始summary，不回写伪造SHA。后续结果审计同时记录源文件SHA256用于核对。部署后没有修改该批实验所导入的训练文件。
