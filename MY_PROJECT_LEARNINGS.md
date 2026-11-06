Bank Churn Prediction using ANN – What I Learned
1. Data Understanding & Cleaning
- Pandas se CSV dataset load aur explore karna seekha.
- Unnecessary columns (RowNumber, CustomerId, Surname) remove kiye.
- Features (X) aur target (y) separate kiye.
- Exited hamara target column tha, jo batata hai customer churn hua ya nahi.
Learning: Model ko sirf relevant features dene chahiye.
2. Train-Test Split & Data Leakage
- train_test_split() se dataset ko training aur testing data mein divide kiya.
- Seekha ki preprocessing se pehle train-test split karna chahiye.
- Agar poore dataset par pehle scaler ya encoder fit kar dein, toh test data ki information training process mein leak ho sakti hai.
Learning: fit_transform() sirf training data par aur transform() test data par use karna chahiye.
3. Feature Encoding & Scaling
- OneHotEncoder se Geography encode ki.
- OrdinalEncoder se Gender encode kiya.
- StandardScaler se numerical features scale kiye.
- ColumnTransformer se saare preprocessing steps combine kiye.
Learning: ANN ko numerical input chahiye. Encoding categorical values ko numerical banati hai aur scaling features ke different ranges ko standardize karti hai.
4. Preprocessor
- ColumnTransformer ko train data par fit_transform() kiya.
- Test data par transform() kiya.
- Preprocessor ko Pickle se save karna seekha.
Learning: New customer ke data par bhi wahi fitted preprocessor use karna hai. Dobara fit nahi karna.
5. ANN Model Building
- TensorFlow aur Keras use kiya.
- Sequential() se ANN model banaya.
- Dense() layers add ki.
- Hidden layers mein ReLU aur output layer mein Sigmoid activation use kiya.
- model.summary() se model architecture aur parameters dekhe.
Learning: Hidden layers patterns learn karti hain aur sigmoid output binary classification ke liye 0–1 ke beech score deta hai.
6. Model Compilation
model.compile() mein teen important cheezein seekhin:
- Adam: Weights aur biases update karta hai.
- Binary Crossentropy: Prediction error calculate karta hai.
- Accuracy: Correct predictions ka proportion batati hai.
Learning: Compile model ko training ke liye configure karta hai; actual training fit() se hoti hai.
7. Model Training & Callbacks
- model.fit() se ANN train kiya.
- Epochs aur batch processing samjhi.
- Training loss aur validation loss ka difference samjha.
- Overfitting aur underfitting samjha.
- EarlyStopping se unnecessary training rokna seekha.
- restore_best_weights=True se best validation performance wale weights restore karna seekha.
Learning: Training data se model learn karta hai aur validation data se uski learning monitor hoti hai.
8. TensorBoard
- TensorBoard callback use kiya.
- logs/fit folder mein event files generate ki.
- Loss, accuracy, validation loss aur validation accuracy ke graphs dekhna seekha.
Learning: TensorBoard model ki training progress visualize karne ke kaam aata hai. Ye trained model save nahi karta.
9. Model Evaluation vs Prediction
Function	Kya kaam karta hai?
model.evaluate(X_test, y_test)	Test data par loss aur accuracy calculate karta hai
model.predict(X_test)	Test data ke liye predictions generate karta hai
model.predict(new_data)	New customer ke liye prediction generate karta hai

Learning: Evaluation model ki performance check karta hai, jabki prediction actual output deta hai.








11. New Customer Prediction
- New customer ki details dictionary mein di.
- pd.DataFrame([new_data]) se dictionary ko DataFrame mein convert kiya.
- Fitted preprocessor se transform() kiya.
- get_feature_names_out() se transformed columns ke names liye.
- Processed data ANN model ko diya.
- Churn probability generate ki.
Learning: New data ke liye training dobara nahi karni padti. Same trained model aur same fitted preprocessor use hote hain.
12. Model Saving & Loading
- model.save() se trained ANN save karna.
- load_model() se ANN load karna.
- pickle.dump() se preprocessor save karna.
- pickle.load() se preprocessor load karna.
Learning: Model aur preprocessor dono save karna zaroori hai, taaki future predictions mein same preprocessing use ho.







# Important Technical Learnings

1. **Data Leakage:** Train-test split preprocessing se pehle karna chahiye. Preprocessor ko sirf training data par fit karna chahiye.

2. **fit_transform vs transform:**
   - `fit_transform()` training data par: parameters learn + transformation.
   - `transform()` test/new data par: learned parameters apply.

3. **Validation vs Test:** Validation data model monitoring aur EarlyStopping ke liye hota hai. Test data final evaluation ke liye untouched rehna chahiye. Agar `X_test` ko validation data banaya hai, toh woh independent final test set nahi raha.

4. **Evaluate vs Predict:**
   - `model.evaluate()` → Loss aur metrics.
   - `model.predict()` → Predictions/probabilities.

5. **Probability Extraction:** `prediction[0][0]` first customer ki first output probability extract karta hai. Ye probability calculate nahi karta.

6. **Threshold:** Sigmoid output ko 0.5 threshold se compare karke class 0 ya 1 decide kiya.

7. **New Data:** New customer ko DataFrame mein convert karke same fitted preprocessor se `transform()` kiya. Model ko retrain nahi kiya.

8. **Model Persistence:** Trained ANN aur fitted preprocessor dono save karna zaroori hai.