<!DOCTYPE html>
<html>
<head>
	<title>AUTHORITY HELP!</title>
</head>

<body style="font-family: helvetica; background-color:#b5c5dd;">
<br>
<br>

<h1 style="text-align: center; size: 40px">"THE AUTHORITIES HAVE BEEN INFORMED OF YOUR LOCATION"</h1>
<br>
<h1 style="text-align: center; size: 40px"><br>"DO NOT PANIC. HELP IS ON THE WAY"</h1>
<br>
<br>
<h2 style="text-align: center; size: 40px">"<a href="index.html">CLICK HERE</a> TO RETURN TO HOMEPAGE"</h2>

<?php
$host = 'civilianserver.mysql.database.azure.com';
$username = 'serveradmin@civilianserver';
$password = 'itsnotpesu123!';
$db_name = 'civiliandb';

//Establishes the connection
$conn = mysqli_init();
mysqli_real_connect($conn, $host, $username, $password, $db_name, 3306);
if (mysqli_connect_errno($conn)) {
die('Failed to connect to MySQL: '.mysqli_connect_error());
}



// Run the create table query
if (mysqli_query($conn, '
CREATE TABLE helpusers (
`Id` INT NOT NULL AUTO_INCREMENT ,
`userlat` REAL NOT NULL ,
`userlon` REAL NOT NULL ,
PRIMARY KEY (`Id`)
);
')) {
//Printf("Table created\n");
}

$username = "ishwar";
$userage = 5;
$userlat = 23.3342;
$userlon = 21.3435;
$userlat = $_GET['latitude'];
$userlon = $_GET['longitude'];



if($stmt = mysqli_prepare($conn, "INSERT INTO helpusers(userlon, userlat) VALUES (?, ?)")) {
mysqli_stmt_bind_param($stmt, 'dd',$userlon,$userlat);
mysqli_stmt_execute($stmt);
//printf("Insert: Affected %d rows\n", mysqli_stmt_affected_rows($stmt));
mysqli_stmt_close($stmt);
}





//Close the connection
mysqli_close($conn);
?>

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
 <h2 style="text-align: center;">"This website was developed Abhilash Balaji and Ishwar Choudhary of PES UNIVERSITY, BANGALORE for the Microsoft Codefundoo++ PROJECT 2018"</h2>


</body>
</html>