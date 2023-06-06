import pandas as pd
import os
# 获取当前脚本所在的目录
dir_path = os.path.dirname(os.path.abspath(__file__))

# 将当前工作目录设置为当前脚本所在的目录
os.chdir(dir_path)

name='false'
root_path='/home/u2022/nfs/waq/Data/Archive'
xlsx_path=os.path.join(root_path,name+'.xlsx')
# false_path=os.path.join(root_path,'false.xlsx')
# 创建ExcelFile对象并打开Excel文件
xls = pd.ExcelFile(xlsx_path)
sheet_names = xls.sheet_names
print(sheet_names)
# 读取Excel文件中的Sheet1和Sheet2工作表
df_list = []
for sheet_name in sheet_names:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
    df['sheet_name']=sheet_name
    df_list.append(df)

# 将多个DataFrame合并到一个DataFrame中，并新增一列来标记每行数据所在的工作表名称
df = pd.concat(df_list, keys=sheet_names)
print(df)

df.to_csv('./data/'+name+'.csv', index=False,encoding='utf-8-sig')