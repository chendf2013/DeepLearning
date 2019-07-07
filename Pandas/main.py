import pandas as pd

input_path = "C:\\Users\\chend\\Desktop\\shaocun\\Input Data.xlsx"
output_path = "C:\\Users\\chend\\Desktop\\shaocun\\Result.xlsx"


def main():
    data = pd.read_excel(input_path)
    sum_voice_traffic = data[["Voice Traffic"]].sum()
    sum_data_traffic = data[["Data Traffic"]].sum()
    print("Voice Traffic列的和是{}".format(str(sum_voice_traffic.values[0])))
    print("Data Traffic列的和是{}".format(str(sum_data_traffic.values[0])))
    data["voice_traffic_and_data_traffic"] = data["Voice Traffic"] + data["Data Traffic"]
    data_res_list = []
    for i in range(0, data.shape[0], 24):
        a = data["voice_traffic_and_data_traffic"][i:i + 24].max()
        if a == 0:
            b = data.iloc[i:i + 1, :]
        else:
            b = data[:][i:i + 24][data['voice_traffic_and_data_traffic'].isin([a])]
        data_res_list.append(b)
    data_res = pd.concat(data_res_list)
    data_res = data_res.iloc[:-1, 0:-1]
    with pd.ExcelWriter(output_path) as writer:
        data_res.to_excel(writer, index=None, sheet_name=str(0))


if __name__ == "__main__":
    main()
