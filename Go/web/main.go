package main

import (
    "fmt"
    "html/template"
    "net/http"
    "os"
	"io"
)

func uploadFile(w http.ResponseWriter, r *http.Request) {
    if r.Method == "GET" {
        // 顯示上傳文件的表單
        t, _ := template.ParseFiles("/Users/chenyenlun/Desktop/Github/PC/Go/web/upload.html")
        t.Execute(w, nil)
    } else if r.Method == "POST" {
        // 處理上傳的文件
        file, handler, err := r.FormFile("file")
        if err != nil {
            fmt.Println(err)
            return
        }
        defer file.Close()

        // 創建一個新文件來保存上傳的文件
        storagePath := "/Users/chenyenlun/Desktop/Github/PC/Go/web/received_files/" + handler.Filename
        dst, err := os.Create(storagePath)
        if err != nil {
            fmt.Println(err)
            return
        }
        defer dst.Close()

        // 將上傳的文件內容複製到新文件中
        if _, err := io.Copy(dst, file); err != nil {
            fmt.Println(err)
            return
        }

        fmt.Fprintf(w, "上傳成功: %s", handler.Filename)
    }
}

func main() {
    http.HandleFunc("/upload", uploadFile)
    http.ListenAndServe(":8080", nil)
}
