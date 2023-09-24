package main

import (
    "fmt"
    "html/template"
    "log"
    "net/http"
    "strings"
    "time"
    "io"
    "os"
    "strconv"
)

func sayhelloName(w http.ResponseWriter, r *http.Request) {
    r.ParseForm()       //解析 url 傳遞的參數，對於 POST 則解析 HTTP 回應內容的主體（request body）
    //注意 : 如果沒有呼叫 ParseForm 方法，下面無法取得表單的資料
    fmt.Println(r.Form) //這些資訊是輸出到伺服器端的列印資訊
    fmt.Println("path", r.URL.Path)
    fmt.Println("scheme", r.URL.Scheme)
    fmt.Println(r.Form["url_long"])
    for k, v := range r.Form {
        fmt.Println("key:", k)
        fmt.Println("val:", strings.Join(v, ""))
    }
    fmt.Fprintf(w, "Hello astaxie!") //這個寫入到 w 的是輸出到客戶端的
}

func login(w http.ResponseWriter, r *http.Request) {
    fmt.Println("method:", r.Method) //取得請求的方法
    if r.Method == "GET" {
        t, _ := template.ParseFiles("/Users/chenyenlun/Desktop/Github/PC/Go/web/login.gtpl")
        log.Println(t.Execute(w, nil))
    } else {
        //請求的是登入資料，那麼執行登入的邏輯判斷
        r.ParseForm()//讀取帳密
        fmt.Println("username:", r.Form["username"])
        fmt.Println("password:", r.Form["password"])
    }
}

func upload(w http.ResponseWriter, r *http.Request) {
    fmt.Println("method:", r.Method) //取得請求的方法
    if r.Method == "GET" {
        crutime := time.Now().Unix()
        h := md5.New()
        io.WriteString(h, strconv.FormatInt(crutime, 10))
        token := fmt.Sprintf("%x", h.Sum(nil))

        t, _ := template.ParseFiles("upload.gtpl")
        t.Execute(w, token)
    } else {
        r.ParseMultipartForm(32 << 20)
        file, handler, err := r.FormFile("uploadfile")
        if err != nil {
            fmt.Println(err)
            return
        }
        defer file.Close()
        fmt.Fprintf(w, "%v", handler.Header)
        f, err := os.OpenFile("./test/"+handler.Filename, os.O_WRONLY|os.O_CREATE, 0666)  // 此處假設當前目錄下已存在 test 目錄
        if err != nil {
            fmt.Println(err)
            return
        }
        defer f.Close()
        io.Copy(f, file)
    }
}

func main() {
    http.HandleFunc("/", sayhelloName)       //設定存取的路由
    http.HandleFunc("/login", login)         //設定存取的路由
    http.HandleFunc("/upload", upload)       // 處理/upload 邏輯
    err := http.ListenAndServe(":9090", nil) //設定監聽的埠
    if err != nil {
        log.Fatal("ListenAndServe: ", err)
    }
}