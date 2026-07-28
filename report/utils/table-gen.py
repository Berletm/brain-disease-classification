import os
from io import TextIOWrapper

def write_header(t: TextIOWrapper, t_caption: str, cols: int) -> None:
    hdr = (
        r"\begin{table*}[t!]" "\n"
        r"\centering" "\n"
        r"\caption{" + t_caption + "}" "\n"
        r"\label{tab:ablation}" "\n"
        r"\small" "\n"
        r"\renewcommand{\arraystretch}{1.15}" "\n"
        r"\begin{tabular}{" + "c" * cols + "}\n"
    )
    t.write(hdr)

def write_tail(t: TextIOWrapper) -> None:
    tail = (
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table*}" "\n"
    )
    t.write(tail)

def write_table(pth:str, name: str, rows: int, cols: int, data: list) -> None:
    with open(os.path.join(pth, name), mode="w+", encoding='utf-8') as t:
        write_header(t, "Результаты абляционного исследования", cols)
        
        header_row = ""
        
        for i in range(cols):
            header_row += r"\textbf{" + f"{data[0][i]}" + "}"
            if i != cols-1:
                header_row += " & "
        
        header_row += " \\\\"
        
        t_start = r"\toprule" + "\n" + header_row + "\n" + r"\midrule" + "\n"
        t.write(t_start)

        for i in range(1, rows):
            for j in range(0, cols):
                end = " & "
                if j == cols - 1:
                    end = " \\\\"
                
                t.write(f"{data[i][j]}" + end)
            t.write("\n")
        write_tail(t)


def main() -> None:
    data = [["Вариант архитектуры", "Accuracy", "Macro Recall", "Macro Precision", "Macro F1"]]
    pths = [
        "/home/berlet/code/brain-disease-classification/models/single_heuristic_result.txt",
        "/home/berlet/code/brain-disease-classification/models/single_mpca_result.txt",
        "/home/berlet/code/brain-disease-classification/models/multi_mean_result.txt",
        "/home/berlet/code/brain-disease-classification/models/multi_cat_result.txt",
        "/home/berlet/code/brain-disease-classification/models/multi_attention_mean_result.txt",
        "/home/berlet/code/brain-disease-classification/models/multi_attention_cat_result.txt"
    ]
    
    names = ["Single Branch (Эвристический срез)", "Single Branch (MPCA)", "Multi Branch (MPCA + Mean)", "Multi Branch (MPCA + Cat)", "Multi Branch (MPCA + MHA + Mean)", "Multi Branch (MPCA + MHA + Cat)"]
    for i, p in enumerate(pths):
        with open(p, mode="r") as f:
            temp = [names[i]]
            if i == len(pths) - 1:
                temp[0] = r"\textbf{" + f"{names[i]}" + "}"
            for row in f:
                mean, std = row.split(":")[1].rstrip().split("+-")
                s = rf"${mean.strip()} \pm {std.strip()}$"
                if i == len(pths) - 1:
                    s = r"$\mathbf{" + rf"{mean.strip()} \pm {std.strip()}" + "}$"
                temp.append(s)
            data.append(temp)
    print(data)
    pth = "/home/berlet/code/brain-disease-classification/report/utils"
    write_table(pth, "ablation-table.tex", len(pths) + 1, len(data[0]), data)
    
if __name__ == "__main__":
    main()