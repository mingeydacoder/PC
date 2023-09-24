<html>
<head>
<title></title>
</head>
<body>
<form action="/login" method="post">
    使用者名稱:<input type="text" name="username">
    密碼:<input type="password" name="password">
    <input type="submit" value="登入">
</form>

<select name="fruit">
<option value="apple">apple</option>
<option value="pear">pear</option>
<option value="banana">banana</option>
</select>

<form enctype="multipart/form-data" action="/upload" method="post">
  <input type="file" name="uploadfile" />
  <input type="hidden" name="token" value="{{.}}"/>
  <input type="submit" value="upload" />
</form>

</body>
</html>