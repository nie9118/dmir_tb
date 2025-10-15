d = {
    "ETTh1": {
        "Nonstationary_Transformer": "    --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --data ETTh1   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --p_hidden_dims 256 256   --p_hidden_layers 2   --d_model 128     ",
        "iTransformer": "    --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --data ETTh1   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7  --d_model 256  --d_ff 256  ",
        "MICN": "   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --data ETTh1   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7 --learning_rate 0.001    ",
        "TimesNet": "   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --data ETTh1   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --d_model 16   --d_ff 32   --top_k 5   ",
        "WITRAN": "   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --data ETTh1   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7  --WITRAN_grid_cols 12  --WITRAN_deal standard --learning_rate 0.001   ",
        "Autoformer": "   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --data ETTh1   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   ",
        "DLinear": "   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --data ETTh1   --features M   --enc_in 7   --dec_in 7   --c_out 7  --learning_rate 0.001   ",
        "Koopa": "    --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --data ETTh1   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --learning_rate 0.001 ",
        "Foil": "    --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --data ETTh1   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --learning_rate 0.001 ",
        "FAN": "    --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --data ETTh1   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --learning_rate 0.001 ",
         "FEDformer": "    --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --data ETTh1   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --learning_rate 0.001 "
    },
    "ETTh2": {
        "Nonstationary_Transformer": "   --root_path ./dataset/ETT-small/   --data_path ETTh2.csv   --data ETTh2   --features M   --e_layers 2   --d_layers 1   --factor 1   --enc_in 7   --dec_in 7   --c_out 7   --p_hidden_dims 256 256   --p_hidden_layers 2  ",
        "iTransformer": "   --root_path ./dataset/ETT-small/   --data_path ETTh2.csv   --data ETTh2   --features M   --e_layers 2   --d_layers 1   --factor 1   --enc_in 7   --dec_in 7   --c_out 7   --d_model 128  --d_ff 128   ",
        "MICN": "   --root_path ./dataset/ETT-small/   --data_path ETTh2.csv   --data ETTh2   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7 --learning_rate 0.001  ",
        "TimesNet": "   --root_path ./dataset/ETT-small/   --data_path ETTh2.csv   --data ETTh2   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --d_model 32   --d_ff 32   --top_k 5   ",
        "WITRAN": "  --root_path ./dataset/ETT-small/   --data_path ETTh2.csv   --data ETTh2   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --WITRAN_grid_cols 12  --WITRAN_deal standard  --learning_rate 0.001  ",
        "Autoformer": "   --root_path ./dataset/ETT-small/   --data_path ETTh2.csv   --data ETTh2   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7    ",
        "DLinear": "   --root_path ./dataset/ETT-small/   --data_path ETTh2.csv   --data ETTh2   --features M   --enc_in 7   --dec_in 7   --c_out 7    --learning_rate 0.001   ",
        "Koopa": "   --root_path ./dataset/ETT-small/   --data_path ETTh2.csv   --data ETTh2   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --learning_rate 0.001    ",
        "Foil": "   --root_path ./dataset/ETT-small/   --data_path ETTh2.csv   --data ETTh2   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --learning_rate 0.001    ",
        "FAN": "   --root_path ./dataset/ETT-small/   --data_path ETTh2.csv   --data ETTh2   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --learning_rate 0.001    ",
        "FEDformer": "   --root_path ./dataset/ETT-small/   --data_path ETTh2.csv   --data ETTh2   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --learning_rate 0.001    "
    },
    "Exchange": {
        "Nonstationary_Transformer": "   --root_path ./dataset/exchange_rate/   --data_path exchange_rate.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --p_hidden_dims 256 256   --p_hidden_layers 2  ",
        "iTransformer": "   --root_path ./dataset/exchange_rate/   --data_path exchange_rate.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8  --d_model 128  --d_ff 128  ",
        "MICN": "   --root_path ./dataset/exchange_rate/   --data_path exchange_rate.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --d_model 64   --d_ff 64   --top_k 5 --learning_rate 0.001     ",
        "TimesNet": "   --root_path ./dataset/exchange_rate/   --data_path exchange_rate.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --d_model 64   --d_ff 64   --top_k 5    ",
        "WITRAN": "  --root_path ./dataset/exchange_rate/   --data_path exchange_rate.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --WITRAN_grid_cols 12  --WITRAN_deal standard --learning_rate 0.001   ",
        "Autoformer": "   --root_path ./dataset/exchange_rate/   --data_path exchange_rate.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8    ",
        "DLinear": "   --root_path ./dataset/exchange_rate/   --data_path exchange_rate.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --learning_rate 0.001    ",
        "Koopa": "   --root_path ./dataset/exchange_rate/   --data_path exchange_rate.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --learning_rate 0.001   ",
        "Foil": "   --root_path ./dataset/exchange_rate/   --data_path exchange_rate.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --learning_rate 0.001   ",
        "FAN": "   --root_path ./dataset/exchange_rate/   --data_path exchange_rate.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --learning_rate 0.001   ",
        "FEDformer": "   --root_path ./dataset/exchange_rate/   --data_path exchange_rate.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --learning_rate 0.001   "

    },
    "Weather": {
        "Nonstationary_Transformer": "   --root_path ./dataset/weather/   --data_path weather.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 21   --dec_in 21   --c_out 21   --train_epochs 3   --p_hidden_dims 256 256   --p_hidden_layers 2  ",
        "iTransformer": "   --root_path ./dataset/weather/   --data_path weather.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 21   --dec_in 21   --c_out 21     --d_model 512   --d_ff 512 ",
        "MICN": "   --root_path ./dataset/weather/   --data_path weather.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 21   --dec_in 21   --c_out 21   --d_model 32   --d_ff 32   --top_k 5   --learning_rate 0.001   ",
        "TimesNet": "   --root_path ./dataset/weather/   --data_path weather.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 21   --dec_in 21   --c_out 21   --d_model 32   --d_ff 32   --top_k 5   ",
        "Autoformer": "   --root_path ./dataset/weather/   --data_path weather.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 21   --dec_in 21   --c_out 21 ",
        "WITRAN": "   --root_path ./dataset/weather/   --data_path weather.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 21   --dec_in 21   --c_out 21  --WITRAN_grid_cols 12  --WITRAN_deal standard  --learning_rate 0.001   ",
        "DLinear": "   --root_path ./dataset/weather/   --data_path weather.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 21   --dec_in 21   --c_out 21  --learning_rate 0.001  ",
        "Koopa": "  --root_path ./dataset/weather/   --data_path weather.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 21   --dec_in 21   --c_out 21  --learning_rate 0.001 ",
        "Foil": "  --root_path ./dataset/weather/   --data_path weather.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 21   --dec_in 21   --c_out 21  --learning_rate 0.001 ",
        "FAN": "  --root_path ./dataset/weather/   --data_path weather.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 21   --dec_in 21   --c_out 21  --learning_rate 0.001 ",
        "FEDformer": "  --root_path ./dataset/weather/   --data_path weather.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 21   --dec_in 21   --c_out 21  --learning_rate 0.001 "
    },
    "ILI": {
        "Nonstationary_Transformer": "   --root_path ./dataset/illness/   --data_path national_illness.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --p_hidden_dims 32 32   --p_hidden_layers 2    ",
        "iTransformer": "   --root_path ./dataset/illness/   --data_path national_illness.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7    --d_model 128   --d_ff 128  ",
        "MICN": "   --root_path ./dataset/illness/   --data_path national_illness.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --d_model 768   --d_ff 768   --top_k 5  --learning_rate 0.001    ",
        "TimesNet": "    --root_path ./dataset/illness/   --data_path national_illness.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --d_model 768   --d_ff 768   --top_k 5    ",
        "Autoformer": "   --root_path ./dataset/illness/   --data_path national_illness.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7    ",
        "WITRAN": "  --root_path ./dataset/illness/   --data_path national_illness.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7  --WITRAN_grid_cols 12  --WITRAN_deal standard  --learning_rate 0.001     ",
        "DLinear": "  --root_path ./dataset/illness/   --data_path national_illness.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --learning_rate 0.001  ",
        "Koopa": "   --root_path ./dataset/illness/   --data_path national_illness.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --learning_rate 0.001    ",
        "Foil": "   --root_path ./dataset/illness/   --data_path national_illness.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --learning_rate 0.001    ",
        "FAN": "   --root_path ./dataset/illness/   --data_path national_illness.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --learning_rate 0.001    ",
        "FEDformer": "   --root_path ./dataset/illness/   --data_path national_illness.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --learning_rate 0.001    "
    },
    "Traffic": {
        "Nonstationary_Transformer": "    --root_path ./dataset/traffic/   --data_path traffic.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 862   --dec_in 862   --c_out 862   --train_epochs 3   --p_hidden_dims 128 128   --p_hidden_layers 2     ",
        "iTransformer": "    --root_path ./dataset/traffic/   --data_path traffic.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 862   --dec_in 862   --c_out 862   --d_model 512   --d_ff 512   ",
        "MICN": "    --root_path ./dataset/traffic/   --data_path traffic.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 862   --dec_in 862   --c_out 862   --d_model 512   --d_ff 512   --top_k 5   --learning_rate 0.001     ",
        "TimesNet": "    --root_path ./dataset/traffic/   --data_path traffic.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 862   --dec_in 862   --c_out 862   --d_model 512   --d_ff 512   --top_k 5      ",
        "Autoformer": "   --root_path ./dataset/traffic/   --data_path traffic.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 862   --dec_in 862   --c_out 862       ",
        "WITRAN": "    --root_path ./dataset/traffic/   --data_path traffic.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 862   --dec_in 862   --c_out 862    --WITRAN_grid_cols 12  --WITRAN_deal standard  --learning_rate 0.001     ",
        "DLinear": "    --root_path ./dataset/traffic/   --data_path traffic.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 862   --dec_in 862   --c_out 862     --learning_rate 0.001  ",
        "Koopa": "   --root_path ./dataset/traffic/   --data_path traffic.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 862   --dec_in 862   --c_out 862   --learning_rate 0.001       ",
        "Foil": "   --root_path ./dataset/traffic/   --data_path traffic.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 862   --dec_in 862   --c_out 862   --learning_rate 0.001       ",
        "FAN": "   --root_path ./dataset/traffic/   --data_path traffic.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 862   --dec_in 862   --c_out 862   --learning_rate 0.001       ",
        "FEDformer": "   --root_path ./dataset/traffic/   --data_path traffic.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 862   --dec_in 862   --c_out 862   --learning_rate 0.001       "
    },
    "ECL": {
        "Nonstationary_Transformer": "    --root_path ./dataset/electricity/   --data_path electricity.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 321   --dec_in 321   --c_out 321   --p_hidden_dims 256 256   --p_hidden_layers 2   --d_model 2048     ",
        "iTransformer": "    --root_path ./dataset/electricity/   --data_path electricity.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 321   --dec_in 321   --c_out 321    --d_model 512   --d_ff 512    ",
        "MICN": "    --root_path ./dataset/electricity/   --data_path electricity.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 321   --dec_in 321   --c_out 321   --d_model 256   --d_ff 512   --top_k 5    --learning_rate 0.001     ",
        "TimesNet": "    --root_path ./dataset/electricity/   --data_path electricity.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 321   --dec_in 321   --c_out 321   --d_model 256   --d_ff 512   --top_k 5          ",
        "Autoformer": "   --root_path ./dataset/electricity/   --data_path electricity.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 321   --dec_in 321   --c_out 321        ",
        "WITRAN": "   --root_path ./dataset/electricity/   --data_path electricity.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 321   --dec_in 321   --c_out 321      --WITRAN_grid_cols 12  --WITRAN_deal standard  --learning_rate 0.001     ",
        "DLinear": "   --root_path ./dataset/electricity/   --data_path electricity.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 321   --dec_in 321   --c_out 321        --learning_rate 0.001  ",
        "Koopa": "    --root_path ./dataset/electricity/   --data_path electricity.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 321   --dec_in 321   --c_out 321   --learning_rate 0.001       ",
        "Foil": "    --root_path ./dataset/electricity/   --data_path electricity.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 321   --dec_in 321   --c_out 321   --learning_rate 0.001       ",
        "FAN": "    --root_path ./dataset/electricity/   --data_path electricity.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 321   --dec_in 321   --c_out 321   --learning_rate 0.001       ",
        "FEDformer": "    --root_path ./dataset/electricity/   --data_path electricity.csv   --data custom   --features M   --e_layers 2   --d_layers 1   --factor 3   --enc_in 321   --dec_in 321   --c_out 321   --learning_rate 0.001       "
    }

}
