package main

import "fmt"

func main() {
    fmt.Printf("Hello world\n")
	aa := 1
	const pi float32 = 3.1415
	fmt.Printf("%d\n",aa)
	fmt.Printf("%f\n",pi)

	const (
		x = iota // x == 0
		y = iota // y == 1
		z  // z == 2
		w        // 常數宣告省略值時，預設和之前一個值的字面相同。這裡隱含的說 w = iota，因此 w == 3。其實上面 y 和 z 可同樣不用"= iota"
	)
	
	const v = iota // 每遇到一個 const 關鍵字，iota 就會重置，此時 v == 0
	
	const (
		h, i, j = iota, iota, iota //h=0,i=0,j=0 iota 在同一行值相同
	)

	const (
		a       = iota //a=0
		b       = "B"
		c       = iota             //c=2
		d, e, f = iota, iota, iota //d=3,e=3,f=3
		g       = iota             //g = 4
	)
	
	fmt.Println(a, b, c, d, e, f, g, h, i, j, x, y, z, w, v)
	
}