YOUR NAME: 陳彥倫
YOUR ID: B10209040

## question 1 answer 
<== 表示給予一個signal變數值，並建立一個constraint。在此範例中，第一個<==將c賦值為a*b之結果，且a輸入需大於2。剩下的兩個表示將a,b指派至poseidon hash函數計算hash值。

## question 2 answer
生成四個 plonk keys: main.plonk.zkey (6996.84KB), main.plonk.vkey.json (2.14KB), main.plonk.sol (25.63KB), main.plonk.html (11111.61KB)。此sol檔案定義了一個稱為 PlonkVerifyer 的合約，其中宣告了多個變數及建立函數。主要的函數 verifyProof 利用多個函數及參數執行如加減法,inverse, Lagrange等數學運算來完成 plonk 驗證邏輯。

## question 3 answer
生成四個 groth16 keys: main.groth16.zkey (133.18KB), main.groth16.vkey.json (3.35KB), main.groth16.sol (11.55KB), main.groth16.html (1961.25KB)。 circuit 表示zkp系統中一連串用來驗證的計算步驟、需要滿足的條件等，其結構有如電路一般。完成 circuit  後執行，會產生 zkey ，可用來驗證是否合法。如 question sheet 中的結果顯示此 prover 以此 zkey 通過了驗證。

## question 4 answer
17499677547561660273017699567908067415377678347145626859540034597523441084050
因e === f, 執行後觀察e之值即為f
## question 5 answer
[hash(b), hash(hash(c)+hash(d))];