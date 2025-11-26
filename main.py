import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import copy

sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

# Индификация устройства с cuda ядрами
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

# Семинар

def ema(y, alpha=0.2):
    """
    Рассчитывает экспоненциальное скользящее среднее, заполняя пропуски.
    """
    s = np.zeros_like(y, dtype=float)

    # Важно: если первый элемент NaN, EMA не сможет стартовать
    # Найдём первый не-NaN элемент для инициализации
    first_valid_index = np.where(~np.isnan(y))[0]
    if len(first_valid_index) == 0:
        # Возвращаем нули, если весь ряд пустой
        return s

    first_idx = first_valid_index[0]
    # Заполняем начальные NaN первым валидным значением
    s[:first_idx] = y[first_idx]
    s[first_idx] = y[first_idx]

    # Рекурсивно вычисляем EMA для всех последующих точек
    for i in range(first_idx + 1, len(y)):
        current_y = y[i]
        if np.isnan(current_y):
            # Если значение пропущено, берём предыдущее сглаженное
            s[i] = s[i - 1]
        else:
            s[i] = alpha * current_y + (1 - alpha) * s[i - 1]

    return s

# CELL_1


train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Работа с временным индексом и базовый EDA

train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])
train_data = train_data.sort_values('timestamp').reset_index(drop=True)

test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
test_data = test_data.sort_values('timestamp').reset_index(drop=True)

print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)
print("\nTrain columns:", train_data.columns.tolist())
print("\nПропуски в train (top 20):")
print(train_data.isnull().sum().sort_values(ascending=False).head(20))
print("\nПропуски в test (top 20):")
print(test_data.isnull().sum().sort_values(ascending=False).head(20))


# Обработка аномалий и пропусков отдельно для train и test

def preprocess_continuous_block(df, continuous_cols):
    for col in continuous_cols:
        if col == 'Temperature':
            df.loc[df[col] <= -10., col] = np.nan
            df.loc[df[col] >= 40., col] = np.nan

    # Заполняем пропуски с помощью EMA для непрерывных признаков
    for col in continuous_cols:
        if df[col].isnull().any():
            df[col + "_old"] = df[col]
            df[col] = ema(df[col].values, alpha=0.3)
            print(f"Пропуски в столбце '{col}' заполнены с помощью EMA.")

    # Не использовать данные будущего
    if continuous_cols:
        df[continuous_cols] = df[continuous_cols].ffill()
    if df.isnull().values.any():
        df.ffill(inplace=True)
    return df

# New logic (OK!)

nim_col = train_data.select_dtypes(include=np.number).columns
bin_col = [
    col for col in nim_col
    if train_data[col].nunique() == 2
    and train_data[col].min() == 0
    and train_data[col].max() == 1
]

con_col = [col for col in nim_col if col not in bin_col]
print("\nНепрерывные признаки в train:", con_col)
train_data = preprocess_continuous_block(train_data, con_col)

num_col = test_data.select_dtypes(include=np.number).columns
binary_cols_test = [
    col for col in num_col
    if test_data[col].nunique() == 2
    and test_data[col].min() == 0
    and test_data[col].max() == 1
]
con_cols_1 = [col for col in num_col if col not in binary_cols_test]
print("\nНепрерывные признаки в test:", con_cols_1)
test_data = preprocess_continuous_block(test_data, con_cols_1)

print("\nВсего пропусков после обработки train:", train_data.isnull().sum().sum())
print("Всего пропусков после обработки test:", test_data.isnull().sum().sum())

# CELL_2

BATCH_SIZE = 128

# New logic (OK!)

target_col = 'Light_Kitchen'
id_col = 'ID'

feature_cols = [
    c for c in train_data.columns
    if c not in [target_col, 'timestamp'] and not c.endswith('_old')
]

feature_cols_test = [
    c for c in test_data.columns
    if c not in [id_col, 'timestamp'] and not c.endswith('_old')
]

print("\nПризнаки train:", feature_cols)
print("Признаки test:", feature_cols_test)

# Нормализация по train

scaler = MinMaxScaler()
train_features = train_data[feature_cols].copy()
test_features = test_data[feature_cols_test].copy()

scaler.fit(train_features.values)

train_features_scaled = pd.DataFrame(
    scaler.transform(train_features.values),
    columns=feature_cols,
    index=train_data.index
)

test_features_scaled = pd.DataFrame(
    scaler.transform(test_features.values),
    columns=feature_cols_test,
    index=test_data.index
)

y_full = train_data[target_col].astype(float).values

# CELL_3

# Сем

def create_sequences_binary(data, seq_length, target_array):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length), :]
        y = target_array[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH_BIN = 120  # Смотрим на 2 минуты истории

# Нарезаем последовательности для всей обучающей истории

X_all, y_all = create_sequences_binary(
    train_features_scaled.values,
    SEQ_LENGTH_BIN,
    y_full
)

print("Форма X_all:", X_all.shape)
print("Форма y_all:", y_all.shape)


X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(
    X_all, y_all, test_size=0.2, shuffle=False
)

#PyTorch-датасеты

train_loader_b = DataLoader(
    TensorDataset(
        torch.from_numpy(X_train_b).float(),
        torch.from_numpy(y_train_b).float()
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_loader_b = DataLoader(
    TensorDataset(
        torch.from_numpy(X_val_b).float(),
        torch.from_numpy(y_val_b).float()
    ),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# CELL_4

def training_loop(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_train_loss += loss.item()

        train_losses.append(running_train_loss / len(train_loader))

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        val_losses.append(running_val_loss / len(val_loader))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f'Эпоха [{epoch+1}/{num_epochs}], '
                f'Train Loss: {train_losses[-1]:.4f}, '
                f'Val Loss: {val_losses[-1]:.4f}'
            )

    return train_losses, val_losses


def plot_losses(train_losses, val_losses, scale='linear'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Кривые обучения')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.yscale(scale)
    plt.legend()
    plt.show()

# CELL_5

# Сем

class BinaryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        h0 = torch.zeros(
            self.lstm.num_layers,
            x.size(0),
            self.lstm.hidden_size
        ).to(x.device)
        c0 = torch.zeros(
            self.lstm.num_layers,
            x.size(0),
            self.lstm.hidden_size
        ).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

input_size_bin = X_train_b.shape[2]
HIDDEN_SIZE = 32
NUM_LAYERS = 2
NUM_EPOCHS = 3

binary_lstm = BinaryLSTM(input_size_bin, HIDDEN_SIZE, NUM_LAYERS).to(device)
criterion_bin = nn.BCEWithLogitsLoss()
optimizer_bin = torch.optim.Adam(binary_lstm.parameters(), lr=0.001)

print("Обучение бинарного классификатора для Light_Kitchen")
bin_train_losses, bin_val_losses = training_loop(
    binary_lstm, train_loader_b, val_loader_b, NUM_EPOCHS, criterion_bin, optimizer_bin, device
)
plot_losses(bin_train_losses, bin_val_losses, scale='linear')

# CELL_5
# сем
def evaluate_model(model, data_loader, device):
    model.eval()
    y_true = []
    y_preds = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Added
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            y_true.extend(labels.cpu().numpy().flatten())
            y_preds.extend(probs.cpu().numpy().flatten())

    y_true = np.array(y_true)
    y_preds = np.array(y_preds)
    print("Оценка завершена!")
    return y_true, y_preds


print("Оценка на валидационной выборке")
val_dataset_full = TensorDataset(
    torch.from_numpy(X_val_b).float(),
    torch.from_numpy(y_val_b).float()
)
val_loader_full = DataLoader(val_dataset_full, batch_size=BATCH_SIZE, shuffle=False)

true_labels_val, probs_val = evaluate_model(binary_lstm, val_loader_full, device)
preds_val = (probs_val > 0.5).astype(int)

print("\nОтчёт по классификации на валидации")
print(
    classification_report(
        true_labels_val,
        preds_val,
        target_names=['Выключен (0)', 'Включен (1)'],
        zero_division=0
    )
)
print(f"\nВсего реальных событий 'Включен': {np.sum(true_labels_val)}")
print(f"Всего предсказано событий 'Включен': {np.sum(preds_val)}")

# CELL_6 Подготовка данных
full_df = pd.concat(
    [
        train_features_scaled.assign(timestamp=train_data['timestamp'], is_test=False),
        test_features_scaled.assign(timestamp=test_data['timestamp'], is_test=True),
    ],
    axis=0
).sort_values('timestamp', kind='mergesort').reset_index(drop=True)
full_array = full_df[feature_cols].values
test_positions = full_df.index[full_df['is_test']].to_numpy()
X_test_sequences = []
valid_mask = []

for pos in test_positions:
    start = pos - SEQ_LENGTH_BIN
    if start < 0:
        valid_mask.append(False)
        X_test_sequences.append(
            np.zeros((SEQ_LENGTH_BIN, full_array.shape[1]), dtype=np.float32)
        )
    else:
        seq = full_array[start:pos, :]
        if seq.shape[0] == SEQ_LENGTH_BIN:
            valid_mask.append(True)
            X_test_sequences.append(seq.astype(np.float32))
        else:
            valid_mask.append(False)
            X_test_sequences.append(
                np.zeros((SEQ_LENGTH_BIN, full_array.shape[1]), dtype=np.float32)
            )

X_test_sequences = np.stack(X_test_sequences, axis=0)
print("Форма X_test_sequences:", X_test_sequences.shape)
print("Количество валидных тестовых окон:", np.sum(valid_mask))

# CELL_7

binary_lstm.eval()
with torch.no_grad():
    test_tensor = torch.from_numpy(X_test_sequences).float().to(device)
    logits = binary_lstm(test_tensor)
    test_probs = torch.sigmoid(logits).cpu().numpy().flatten()

valid_mask_arr = np.array(valid_mask, dtype=bool)
valid_probs = test_probs[valid_mask_arr]
if valid_probs.size > 0:
    fill_value = valid_probs.mean()
else:
    fill_value = 0.0

test_probs_filled = test_probs.copy()
test_probs_filled[~valid_mask_arr] = fill_value

test_preds = (test_probs_filled > 0.5).astype(int)

# CELL_8

if 'ID' in test_data.columns:
    submission = pd.DataFrame({
        'ID': test_data['ID'].values,
        'Light_Kitchen': test_preds.astype(int)
    })
else:
    submission = pd.DataFrame({
        'ID': np.arange(len(test_preds)),
        'Light_Kitchen': test_preds.astype(int)
    })

print(submission.head())
submission.to_csv('submission.csv', index=False)
