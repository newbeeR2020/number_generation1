import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---------------------------
# モデル定義
# ---------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        # ラベル情報を低次元に埋め込む（one-hot から16次元）
        self.label_embedding = nn.Linear(num_classes, 16)
        # 画像（28*28）とラベル埋め込み（16）の結合 -> 128次元の隠れ層へ
        self.fc_hidden = nn.Linear(28 * 28 + 16, 128)
        # 潜在変数の平均と対数分散を算出
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, image, label):
        # 画像をフラット化： (B, 1, 28, 28) -> (B, 784)
        flattened_image = image.view(image.size(0), -1)
        # ラベルを one-hot 表現に変換： (B) -> (B, num_classes)
        label_one_hot = F.one_hot(label, num_classes=10).float()
        # ラベルの線形変換（埋め込み層）
        label_embed_linear = self.label_embedding(label_one_hot)
        # 活性化関数適用
        label_embedding = F.relu(label_embed_linear)
        # 画像情報とラベル埋め込みの結合
        concatenated_input = torch.cat([flattened_image, label_embedding], dim=1)
        # 隠れ層の線形変換→ 活性化関数適用
        hidden_activation = F.relu(self.fc_hidden(concatenated_input))
        # 潜在変数の平均と対数分散の計算
        mu = self.fc_mu(hidden_activation)
        logvar = self.fc_logvar(hidden_activation)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        # ラベルの埋め込み層： one-hot から16次元へ
        self.label_embedding = nn.Linear(num_classes, 16)
        # 潜在変数とラベル埋め込みの結合 -> 128次元の隠れ層へ
        self.fc_hidden = nn.Linear(latent_dim + 16, 128)
        # 128次元の隠れ層から最終出力（28*28=784）への線形変換
        self.fc_out = nn.Linear(128, 28 * 28)

    def forward(self, latent_vector, label):
        # ラベルを one-hot 表現に変換
        label_one_hot = F.one_hot(label, num_classes=10).float()
        # ラベル埋め込みの線形変換→ 活性化関数適用
        label_embedding = F.relu(self.label_embedding(label_one_hot))
        # 潜在変数とラベル埋め込みの結合
        concatenated_latent = torch.cat([latent_vector, label_embedding], dim=1)
        # 隠れ層への線形変換と活性化関数適用
        hidden_activation = F.relu(self.fc_hidden(concatenated_latent))
        # 出力層の線形変換
        output_linear = self.fc_out(hidden_activation)
        # シグモイドを適用して [0,1] の値に変換し、画像形式に reshape
        reconstructed_image = torch.sigmoid(output_linear).view(-1, 1, 28, 28)
        return reconstructed_image

class CVAE(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.encoder = Encoder(latent_dim, num_classes)
        self.decoder = Decoder(latent_dim, num_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, image, label):
        mu, logvar = self.encoder(image, label)
        latent_vector = self.reparameterize(mu, logvar)
        reconstructed_image = self.decoder(latent_vector, label)
        return reconstructed_image, mu, logvar

# ---------------------------
# デバイス設定
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# モデルのロード
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path="model.pth"):
    model = CVAE(latent_dim=3, num_classes=10).to(device)
    # 保存済みのパラメータをロード
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# ---------------------------
# Streamlitによるユーザー入力受付と出力表示
# ---------------------------
st.title("CVAE Digit Generator")

st.write("0～9の数字を選んで、対応する画像を生成します。")

# 数字入力
digit = st.number_input("生成したい数字を入力してください (0～9)", min_value=0, max_value=9, value=5, step=1)

# 生成ボタン
if st.button("生成"):
    # 潜在変数 z を標準正規分布からサンプリング（1サンプル）
    z_random_vector = torch.randn(1, 3).to(device)
    # クラスラベル作成
    label = torch.tensor([digit], dtype=torch.long, device=device)
    # 推論モードで画像生成（Decoderのみを利用）
    with torch.no_grad():
        generated_tensor = model.decoder(z_random_vector, label)
    # tensorをnumpy配列に変換（グレースケール画像）
    generated_image = generated_tensor.squeeze().cpu().numpy()
    # 画像表示
    st.image(generated_image, caption=f"Generated digit: {digit}", width=280)
