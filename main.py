import os
from preprocess import preprocess_data
from TransformerBlock import TransformerBlock
from evaluation import evaluate_and_display_model_performance
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 定义路径常量
DATA_PATH = "./data/data.csv"
TOKENIZER_PATH = "./data/feature.pickle"
PREPROCESSED_DATA_PATH = "./data/preprocessed_data.pickle"
MODEL_SAVE_PATH = "./model/best_model.h5"

def main():
    # 检查路径是否存在
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"The data file was not found at: {DATA_PATH}")
    if not os.path.exists(os.path.dirname(TOKENIZER_PATH)):
        raise FileNotFoundError(f"The directory for the tokenizer path does not exist: {os.path.dirname(TOKENIZER_PATH)}")
    if not os.path.exists(os.path.dirname(PREPROCESSED_DATA_PATH)):
        raise FileNotFoundError(f"The directory for the preprocessed data path does not exist: {os.path.dirname(PREPROCESSED_DATA_PATH)}")
    if not os.path.exists(os.path.dirname(MODEL_SAVE_PATH)):
        raise FileNotFoundError(f"The directory for the model save path does not exist: {os.path.dirname(MODEL_SAVE_PATH)}")

    # 数据预处理
    tokenizer, X_train, X_test, y_train, y_test = preprocess_data(DATA_PATH, TOKENIZER_PATH, PREPROCESSED_DATA_PATH)

    # 模型超参数
    embed_dim = 128
    num_heads = 8
    ff_dim = 1024
    max_words = 10000
    max_len = 1000

    # 构建模型
    model = Sequential([
        Embedding(max_words, embed_dim, input_length=max_len),
        TransformerBlock(embed_dim, num_heads, ff_dim),
        Conv1D(64, 5, activation='relu'),
        MaxPooling1D(pool_size=4),
        Flatten(),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])

    # 设置学习率调度
    lr_schedule = schedules.ExponentialDecay(0.0001, decay_steps=10000, decay_rate=0.9, staircase=True)
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # 早停和检查点回调
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True)

    # 训练模型
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test), callbacks=[early_stopping, model_checkpoint])

    # 模型评估
    y_pred_test_probs = model.predict(X_test).ravel()
    evaluate_and_display_model_performance(y_test, y_pred_test_probs)

if __name__ == "__main__":
    main()