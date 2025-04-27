import pkuseg

seg = pkuseg.pkuseg()
input_file = 'data/test_final.txt'
output_file = 'cws_result.txt'

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open(output_file, 'w', encoding='utf-8') as f_out:
    for line in lines:
        line = line.strip()
        if not line:
            f_out.write('\n')
            continue
        words = seg.cut(line)
        f_out.write(' '.join(words) + '\n')

print('分词完成，输出到', output_file)