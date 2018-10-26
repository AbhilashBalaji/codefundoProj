<!DOCTYPE html>
<html>
<head>
	<title>DETAILS SERVER</title>
</head>
<body style="font-family: helvetica; background-color:#b5c5dd;">


<h1 style="text-align: center; size: 40px"><br>"DETAILS RECORDED!"</h1>
<h1 style="text-align: center;"><br>"TO ENTER INFORMATION ABOUT RELATIVES  <a href="civiliandetails.html">CLICK HERE"</a></h1>


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
CREATE TABLE userdetails (
`Id` INT NOT NULL AUTO_INCREMENT ,
`username` CHAR(50),
`userage` INT NOT NULL ,
`userlat` REAL NOT NULL ,
`userlon` REAL NOT NULL ,
PRIMARY KEY (`Id`)
);
')) {
//Printf("Table created\n");
}


$username = $_POST['username'];
$userage = $_POST['userage'];
$userlat = $_POST['latitude'];
$userlon = $_POST['longitude'];

//echo $username;
// echo $userage;
// echo $userlat;
// echo $userlon;

if($stmt = mysqli_prepare($conn, "INSERT INTO userdetails(username, userlon, userlat, userage) VALUES (?, ?, ?, ?)")) {
mysqli_stmt_bind_param($stmt, 'ssdd', $username, $userlon,$userlat,$userage);
mysqli_stmt_execute($stmt);
//printf("Insert: Affected %d rows\n", mysqli_stmt_affected_rows($stmt));
mysqli_stmt_close($stmt);
}




//Close the connection
mysqli_close($conn);
?>
</body>
</html>