d = {
    "ETTh1": {
        "96": "python run_nsts.py --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --data ETTh1   --features M     --enc_in 7   --dec_in 7   --c_out 7"
              "  --seq_len 36 --label_len 0  --pred_len  12  --model NSTS  --zd_dim 3  --hidden_dim 384  --hidden_layers 2   --dropout 0   --activation relu   --learning_rate 0.001  --is_bn --zd_kl_weight 0.0001 --zc_kl_weight 0.0001  --hmm_weight 0.0001 ",
        "192": "python run_nsts.py --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --data ETTh1   --features M     --enc_in 7   --dec_in 7   --c_out 7"
              "  --seq_len 72 --label_len 0  --pred_len  24  --model NSTS  --zd_dim 3  --hidden_dim 512  --hidden_layers 2   --dropout 0.5   --activation relu  --learning_rate 0.001  ",
        "336": "python run_nsts.py --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --data ETTh1   --features M     --enc_in 7   --dec_in 7   --c_out 7"
              "  --seq_len 144 --label_len 0  --pred_len  48  --model NSTS  --zd_dim 3  --hidden_dim 512  --hidden_layers 2   --dropout 0.5   --activation relu  --learning_rate 0.001  ",
        "720": "python run_nsts.py --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --data ETTh1   --features M     --enc_in 7   --dec_in 7   --c_out 7"
              "  --seq_len 216 --label_len 0  --pred_len  72  --model NSTS  --zd_dim 3  --hidden_dim 128  --hidden_layers 1   --dropout 0   --activation relu  --learning_rate 0.001  ",

    },
 "ETTm1": {
        "96": "python run_nsts.py --root_path ./dataset/ETT-small/   --data_path ETTm1.csv   --data ETTm1   --features M     --enc_in 7   --dec_in 7   --c_out 7"
              "  --seq_len 96 --label_len 0  --pred_len  48  --model NSTS  --zd_dim 3  --hidden_dim 384  --hidden_layers 2   --dropout 0   --activation relu   --learning_rate 0.001  --is_bn --zd_kl_weight 0.0001 --zc_kl_weight 0.0001  --hmm_weight 0.0001 ",
        "192": "python run_nsts.py --root_path ./dataset/ETT-small/   --data_path ETTm1.csv   --data ETTm1   --features M     --enc_in 7   --dec_in 7   --c_out 7"
              "  --seq_len 192 --label_len 0  --pred_len  96  --model NSTS  --zd_dim 3  --hidden_dim 512  --hidden_layers 2   --dropout 0.5   --activation relu  --learning_rate 0.001  ",
        "336": "python run_nsts.py --root_path ./dataset/ETT-small/   --data_path ETTm1.csv   --data ETTm1   --features M     --enc_in 7   --dec_in 7   --c_out 7"
              "  --seq_len 288 --label_len 0  --pred_len  144 --model NSTS  --zd_dim 3  --hidden_dim 512  --hidden_layers 2   --dropout 0.5   --activation relu  --learning_rate 0.001  ",
        "720": "python run_nsts.py --root_path ./dataset/ETT-small/   --data_path ETTm1.csv   --data ETTm1   --features M     --enc_in 7   --dec_in 7   --c_out 7"
              "  --seq_len 384 --label_len 0  --pred_len  192  --model NSTS  --zd_dim 3  --hidden_dim 128  --hidden_layers 1   --dropout 0   --activation relu  --learning_rate 0.001  ",

    },

 "ETTm2": {
         "96": "python run_nsts.py --root_path ./dataset/ETT-small/   --data_path ETTm2.csv   --data ETTm2   --features M     --enc_in 7   --dec_in 7   --c_out 7"
              "  --seq_len 96 --label_len 0  --pred_len  48  --model NSTS  --zd_dim 3  --hidden_dim 384  --hidden_layers 2   --dropout 0   --activation relu   --learning_rate 0.001  --is_bn --zd_kl_weight 0.0001 --zc_kl_weight 0.0001  --hmm_weight 0.0001 ",
        "192": "python run_nsts.py --root_path ./dataset/ETT-small/   --data_path ETTm2.csv   --data ETTm2   --features M     --enc_in 7   --dec_in 7   --c_out 7"
              "  --seq_len 192 --label_len 0  --pred_len  96  --model NSTS  --zd_dim 3  --hidden_dim 512  --hidden_layers 2   --dropout 0.5   --activation relu  --learning_rate 0.001  ",
        "336": "python run_nsts.py --root_path ./dataset/ETT-small/   --data_path ETTm2.csv   --data ETTm2   --features M     --enc_in 7   --dec_in 7   --c_out 7"
              "  --seq_len 288 --label_len 0  --pred_len  144  --model NSTS  --zd_dim 3  --hidden_dim 512  --hidden_layers 2   --dropout 0.5   --activation relu  --learning_rate 0.001  ",
        "720": "python run_nsts.py --root_path ./dataset/ETT-small/   --data_path ETTm2.csv   --data ETTm2   --features M     --enc_in 7   --dec_in 7   --c_out 7"
              "  --seq_len 384 --label_len 0  --pred_len  192  --model NSTS  --zd_dim 3  --hidden_dim 128  --hidden_layers 1   --dropout 0   --activation relu  --learning_rate 0.001  ",

    },

    "ETTh2": {
        "96": "python run_nsts.py --root_path ./dataset/ETT-small/   --data_path ETTh2.csv   --data ETTh2   --features M     --enc_in 7   --dec_in 7   --c_out 7 "
              "  --seq_len 36 --label_len 0  --pred_len  12  --model NSTS  --zd_dim 3  --hidden_dim 512  --hidden_layers 2   --dropout 0   --activation ide  --learning_rate 0.001  ",

        "192": "python run_nsts.py --root_path ./dataset/ETT-small/   --data_path ETTh2.csv   --data ETTh2   --features M     --enc_in 7   --dec_in 7   --c_out 7"
              "  --seq_len 72 --label_len 0  --pred_len  24  --model NSTS  --zd_dim 3  --hidden_dim 512  --hidden_layers 2   --dropout 0   --activation ide  --learning_rate 0.001  ",
        "336": "python run_nsts.py --root_path ./dataset/ETT-small/   --data_path ETTh2.csv   --data ETTh2   --features M     --enc_in 7   --dec_in 7   --c_out 7"
              "  --seq_len 144 --label_len 0  --pred_len  48  --model NSTS  --zd_dim 3  --hidden_dim 512  --hidden_layers 2   --dropout 0   --activation ide  --learning_rate 0.001  ",
        "720": "python run_nsts.py --root_path ./dataset/ETT-small/   --data_path ETTh2.csv   --data ETTh2   --features M     --enc_in 7   --dec_in 7   --c_out 7"
              "  --seq_len 216 --label_len 0  --pred_len  72  --model NSTS  --zd_dim 3  --hidden_dim 256  --hidden_layers 2   --dropout 0   --activation ide  --learning_rate 0.001  "
    },
    "Exchange": {
        "96": "python run_nsts.py --root_path ./dataset/exchange_rate/  --data_path exchange_rate.csv    --data custom    --features M     --enc_in 8  --dec_in 8   --c_out 8 "
              "  --seq_len 36 --label_len 0  --pred_len  12  --model NSTS  --zd_dim 3  --hidden_dim 512  --hidden_layers 2   --dropout 0.5   --activation ide  --learning_rate 0.001  ",
        "192": "python run_nsts.py --root_path ./dataset/exchange_rate/  --data_path exchange_rate.csv    --data custom    --features M     --enc_in 8  --dec_in 8   --c_out 8 "
              "  --seq_len 72 --label_len 0  --pred_len  24  --model NSTS  --zd_dim 3  --hidden_dim 512  --hidden_layers 2   --dropout 0.5   --activation ide  --learning_rate 0.001 ",
        "336": "python run_nsts.py --root_path ./dataset/exchange_rate/  --data_path exchange_rate.csv    --data custom    --features M     --enc_in 8  --dec_in 8   --c_out 8 "
              "  --seq_len 144 --label_len 0  --pred_len  48  --model NSTS  --zd_dim 3  --hidden_dim 512  --hidden_layers 2   --dropout 0.5   --activation ide  --learning_rate 0.001 ",
        "720": "python run_nsts.py --root_path ./dataset/exchange_rate/  --data_path exchange_rate.csv    --data custom    --features M     --enc_in 8  --dec_in 8   --c_out 8 "
              "  --seq_len 216 --label_len 0  --pred_len  72  --model NSTS  --zd_dim 3  --hidden_dim 128  --hidden_layers 2   --dropout 0.5  --activation ide  --learning_rate 0.001 ",

    },
    "ILI": {
        "24": "python run_nsts.py --root_path ./dataset/illness/   --data_path national_illness.csv   --data custom    --features M     --enc_in 7   --dec_in 7   --c_out 7 "
              "  --seq_len 36 --label_len 0  --pred_len  12  --model NSTS  --zd_dim 3  --hidden_dim 512  --hidden_layers 3   --dropout 0   --activation relu  --is_bn  --learning_rate 0.001   ",
        "36": "python run_nsts.py --root_path ./dataset/illness/   --data_path national_illness.csv   --data custom    --features M     --enc_in 7   --dec_in 7   --c_out 7  "
              "  --seq_len 72 --label_len 0  --pred_len  24  --model NSTS  --zd_dim 3  --hidden_dim 512  --hidden_layers 3   --dropout 0   --activation relu --is_bn   --learning_rate 0.001  ",
        "48": "python run_nsts.py --root_path ./dataset/illness/   --data_path national_illness.csv   --data custom    --features M     --enc_in 7   --dec_in 7   --c_out 7  "
              "  --seq_len 108 --label_len 0  --pred_len  36  --model NSTS  --zd_dim 3   --hidden_dim 512  --hidden_layers 3  --dropout 0   --activation relu  --is_bn  --learning_rate 0.001  ",

        "60": "python run_nsts.py --root_path ./dataset/illness/   --data_path national_illness.csv   --data custom    --features M     --enc_in 7   --dec_in 7   --c_out 7  "
              "  --seq_len 144 --label_len 0  --pred_len  48  --model NSTS  --zd_dim 3  --hidden_dim 640  --hidden_layers 2  --dropout 0.2  --activation relu   --learning_rate 0.001   --is_bn --zd_kl_weight 0.0001 --zc_kl_weight 0.0001  --hmm_weight 0.0001  ",

    },
    "Weather": {
        "96": "python run_nsts.py --root_path ./dataset/weather/   --data_path weather.csv  --data custom    --features M     --enc_in 21   --dec_in 21   --c_out 21 "
              "  --seq_len 36 --label_len 0  --pred_len  12  --model NSTS  --zd_dim 3  --hidden_dim 192  --hidden_layers 3   --dropout 0   --activation relu   --learning_rate 0.001  --is_bn --zd_kl_weight 0.0001 --zc_kl_weight 0.0001  --hmm_weight 0.0001  ",
        "192": "python run_nsts.py --root_path ./dataset/weather/   --data_path weather.csv  --data custom    --features M     --enc_in 21   --dec_in 21   --c_out 21   "
              "  --seq_len 72 --label_len 0  --pred_len  24  --model NSTS  --zd_dim 3  --hidden_dim 384  --hidden_layers 3   --dropout 0   --activation relu   --learning_rate 0.001    --is_bn --zd_kl_weight 0.0001 --zc_kl_weight 0.0001  --hmm_weight 0.0001  ",
        "336": "python run_nsts.py --root_path ./dataset/weather/   --data_path weather.csv  --data custom    --features M     --enc_in 21   --dec_in 21   --c_out 21  "
              "  --seq_len 144 --label_len 0  --pred_len  48  --model NSTS  --zd_dim 3  --hidden_dim 192  --hidden_layers 3   --dropout 0   --activation relu   --learning_rate 0.001  --is_bn --zd_kl_weight 0.0001 --zc_kl_weight 0.0001  --hmm_weight 0.0001   ",

        "720": "python run_nsts.py --root_path ./dataset/weather/   --data_path weather.csv  --data custom    --features M     --enc_in 21   --dec_in 21   --c_out 21  "
              "   --seq_len 216 --label_len 0  --pred_len  72  --model NSTS  --zd_dim 3  --hidden_dim 384  --hidden_layers 3   --dropout 0   --activation relu   --learning_rate 0.001   --is_bn --zd_kl_weight 0.0001 --zc_kl_weight 0.0001  --hmm_weight 0.0001 ",

    },
    "ECL": {
        "96": "python run_nsts.py    --root_path ./dataset/electricity/   --data_path electricity.csv   --data custom   --features M      --enc_in 321   --dec_in 321   --c_out 321 "
              "  --seq_len 36 --label_len 0  --pred_len  12  --model NSTS  --zd_dim 3  --hidden_dim 512  --hidden_layers 3   --dropout 0   --activation relu   --learning_rate 0.001  --is_bn --zd_kl_weight 0.000001 --zc_kl_weight 0.000001  --hmm_weight 0.000001  ",
        "192": "python run_nsts.py    --root_path ./dataset/electricity/   --data_path electricity.csv   --data custom   --features M      --enc_in 321   --dec_in 321   --c_out 321  "
              "  --seq_len 72 --label_len 0  --pred_len  24  --model NSTS  --zd_dim 3  --hidden_dim 512  --hidden_layers 3   --dropout 0   --activation relu   --learning_rate 0.001    --is_bn --zd_kl_weight 0.000001 --zc_kl_weight 0.000001  --hmm_weight 0.000001  ",
        "336": "python run_nsts.py    --root_path ./dataset/electricity/   --data_path electricity.csv   --data custom   --features M      --enc_in 321   --dec_in 321   --c_out 321  "
              "  --seq_len 144 --label_len 0  --pred_len  48  --model NSTS  --zd_dim 3  --hidden_dim 512  --hidden_layers 3   --dropout 0   --activation relu   --learning_rate 0.001  --is_bn --zd_kl_weight 0.000001 --zc_kl_weight 0.000001  --hmm_weight 0.000001  ",

        "720": "python run_nsts.py    --root_path ./dataset/electricity/   --data_path electricity.csv   --data custom   --features M      --enc_in 321   --dec_in 321   --c_out 321  "
              "   --seq_len 216 --label_len 0  --pred_len  72  --model NSTS  --zd_dim 3  --hidden_dim 512  --hidden_layers 2   --dropout 0   --activation relu   --learning_rate 0.001   --is_bn --zd_kl_weight 0.000001 --zc_kl_weight 0.000001  --hmm_weight 0.000001 ",

    },
    "Traffic": {
        "96": "python run_nsts.py --root_path ./dataset/traffic/   --data_path traffic.csv --data custom    --features M     --enc_in 862  --dec_in 862  --c_out 862 "
              "  --seq_len 36 --label_len 0  --pred_len  12  --model NSTS  --zd_dim 3  --hidden_dim 256  --hidden_layers 2   --dropout 0   --activation relu   --learning_rate 0.001  --is_bn   --rec_weight 1 --zd_kl_weight 0.000001 --zc_kl_weight 0.000001  --hmm_weight 0.000001  ",
        "192": "python run_nsts.py --root_path ./dataset/traffic/   --data_path traffic.csv --data custom    --features M     --enc_in 862  --dec_in 862  --c_out 862  "
              "  --seq_len 72 --label_len 0  --pred_len  24  --model NSTS  --zd_dim 1  --hidden_dim 384  --hidden_layers 2   --dropout 0   --activation relu   --learning_rate 0.001   --rec_weight 1 --zd_kl_weight 0.000001 --zc_kl_weight 0.000001  --hmm_weight 0.000001  ",
        "336": "python run_nsts.py --root_path ./dataset/traffic/   --data_path traffic.csv --data custom    --features M     --enc_in 862  --dec_in 862  --c_out 862 "
              "  --seq_len 144 --label_len 0  --pred_len  48  --model NSTS  --zd_dim 1  --hidden_dim 384  --hidden_layers 2   --dropout 0   --activation relu   --learning_rate 0.001    --rec_weight 1 --zd_kl_weight 0.000001 --zc_kl_weight 0.000001  --hmm_weight 0.000001   ",

        "720": "python run_nsts.py --root_path ./dataset/traffic/   --data_path traffic.csv --data custom    --features M     --enc_in 862  --dec_in 862  --c_out 862"
              "   --seq_len 216 --label_len 0  --pred_len  72  --model NSTS  --zd_dim 1  --hidden_dim 1024  --hidden_layers 2   --dropout 0   --activation relu   --learning_rate 0.001   --rec_weight 1   --zd_kl_weight 0.000001 --zc_kl_weight 0.000001  --hmm_weight 0.000001 ",

    }


}
