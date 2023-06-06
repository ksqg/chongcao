import pandas as pd
import os
# 获取当前脚本所在的目录
dir_path = os.path.dirname(os.path.abspath(__file__))

# 将当前工作目录设置为当前脚本所在的目录
os.chdir(dir_path)

f2_path='/home/u2022/nfs/waq/虫草/data/true.csv'
df = pd.read_csv(f2_path)

print(df)

df['label_part'] = df["0"].apply(lambda x: x[-1])
df['label_phase'] = df["0"].apply(lambda x: x[-2])
print(df)
df.to_csv('./data/true_label.csv', index=False,encoding='utf-8-sig')
