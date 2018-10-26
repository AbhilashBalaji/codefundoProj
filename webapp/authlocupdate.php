<!DOCTYPE html>
<html>
<head>
	<title>LOCATION UPDATED</title>
</head>
<body>

<?php 

$lat1 = $_POST['lat1'];
$lat2 = $_POST['lat2'];
$lat3 = $_POST['lat3'];
$lat4 = $_POST['lat4'];
$lat5 = $_POST['lat5'];
$lon1 = $_POST['lon1'];
$lon2 = $_POST['lon2'];
$lon3 = $_POST['lon3'];
$lon4 = $_POST['lon4'];
$lon5 = $_POST['lon5'];

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
CREATE TABLE safelocations (
`Id` INT NOT NULL AUTO_INCREMENT ,
`safelat` REAL NOT NULL ,
`safelon` REAL NOT NULL ,
PRIMARY KEY (`Id`)
);
')) {
Printf("Table created\n");
}


if($stmt = mysqli_prepare($conn, "INSERT INTO safelocations(safelon, safelat) VALUES (?, ?)")) {
mysqli_stmt_bind_param($stmt, 'dd',$lon1,$lat1);
mysqli_stmt_execute($stmt);
printf("Insert: Affected %d rows\n", mysqli_stmt_affected_rows($stmt));
mysqli_stmt_close($stmt);
}
if($stmt = mysqli_prepare($conn, "INSERT INTO safelocations(safelon, safelat) VALUES (?, ?)")) {
mysqli_stmt_bind_param($stmt, 'dd',$lon2,$lat2);
mysqli_stmt_execute($stmt);
printf("Insert: Affected %d rows\n", mysqli_stmt_affected_rows($stmt));
mysqli_stmt_close($stmt);
}
if($stmt = mysqli_prepare($conn, "INSERT INTO safelocations(safelon, safelat) VALUES (?, ?)")) {
mysqli_stmt_bind_param($stmt, 'dd',$lon3,$lat3);
mysqli_stmt_execute($stmt);
printf("Insert: Affected %d rows\n", mysqli_stmt_affected_rows($stmt));
mysqli_stmt_close($stmt);
}
if($stmt = mysqli_prepare($conn, "INSERT INTO safelocations(safelon, safelat) VALUES (?, ?)")) {
mysqli_stmt_bind_param($stmt, 'dd',$lon4,$lat4);
mysqli_stmt_execute($stmt);
printf("Insert: Affected %d rows\n", mysqli_stmt_affected_rows($stmt));
mysqli_stmt_close($stmt);
}
if($stmt = mysqli_prepare($conn, "INSERT INTO safelocations(safelon, safelat) VALUES (?, ?)")) {
mysqli_stmt_bind_param($stmt, 'dd',$lon5,$lat5);
mysqli_stmt_execute($stmt);
printf("Insert: Affected %d rows\n", mysqli_stmt_affected_rows($stmt));
mysqli_stmt_close($stmt);
}


?>

</body>
</html>