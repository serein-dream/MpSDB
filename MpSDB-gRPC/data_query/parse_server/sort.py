# 读取文本内容
with open('/home/maxvyang01/FaaS_FL2023/SecureDatabase/data_query/parse_server/test_if_multi.txt', 'r') as file:
    lines = file.readlines()

# 解析每一行为字典
data = {}
for line in lines:
    if ':' in line:
        key, *value = line.strip().split(':')
        if value:
            data[key] = float(value[0])

# 按数值升序排序并输出
sorted_data = dict(sorted(data.items(), key=lambda item: item[1]))

with open('/home/maxvyang01/FaaS_FL2023/SecureDatabase/data_query/parse_server/test_if_multi_sort.txt', 'w') as output_file:
    for key, value in sorted_data.items():
        output_file.write(f"{key}:{value}\n")
