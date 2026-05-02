import os

def extract_from_tomtom_tsv(input_file, output_file):
    seen_sequences = set()
    motif_counter = 1
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        # 跳过表头
        next(fin)
        
        for line in fin:
            if not line.strip() or line.startswith('#'):
                continue
                
            columns = line.split('\t')
            sequence = columns[0].strip()
            
            if sequence.isalpha() and sequence.isupper():
                if sequence not in seen_sequences:
                    fout.write(f">Motif_{motif_counter}\n")
                    fout.write(f"{sequence}\n")
                    seen_sequences.add(sequence)
                    motif_counter += 1

    print(f"✅ 成功提取 {len(seen_sequences)} 个 Motif！")
    print(f"📁 结果已保存至: {output_file}")

if __name__ == "__main__":
    # 自动获取当前 python 脚本所在的绝对目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 将目录与文件名拼接，确保路径绝对正确
    input_filename = os.path.join(script_dir, 'tomtom.tsv')
    output_filename = os.path.join(script_dir, 'my_motifs.fasta')
    
    if os.path.exists(input_filename):
        extract_from_tomtom_tsv(input_filename, output_filename)
    else:
        print(f"❌ 错误：依然找不到文件。")
        print(f"程序试图在以下路径寻找: {input_filename}")
        print("请确认 tomtom.tsv 文件是否真的存在于该位置。")