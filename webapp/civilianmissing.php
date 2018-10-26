<!DOCTYPE html>
<html>
<head>
	<title>MISSING PERSON!</title>
	<style type="text/css">
	 .button {
    background-color: #f7f9fc; 
    border: none;
    color: black;
    padding: 20px 30px;
    text-align: center;
    text-decoration: none;
    margin: auto;
    margin-top: 50px;

    display: block;
    justify-content: center;
    font-size: 30px;
    position: center;
    border-radius: 6px;
    border: 2px solid black;
    font-family: helvetica;
    }   
    body{
        background-color:#b5c5dd;
    }
    </style>
</head>
<body style="font-family: helvetica; background-color:#b5c5dd;" >

 <h1 style="text-align: center;"><br>"STATUS OF THE CIVILIAN"</h1>
 <h1 style="text-align: center;" id="status"><br>"CIVILIAN FOUND!"</h1> 
 <br>
 <h2 style="text-align: center;">CIVILIAN COORDINATES:<h2>
 <h2 style="text-align: center;" id="civlat"><br></h2> 
 <h2 style="text-align: center;" id="civlon"><br></h2> 
 
 <br>
 <br>

<?php 

//fetching data from server
$host = 'civilianserver.mysql.database.azure.com';
$username = 'serveradmin@civilianserver';
$password = 'itsnotpesu123!';
$db_name = 'civiliandb';


$civname = $_POST['username'];
$civage = $_POST['userage'];

//Establishes the connection
$conn = mysqli_init();
mysqli_real_connect($conn, $host, $username, $password, $db_name, 3306);
if (mysqli_connect_errno($conn)) {
die('Failed to connect to MySQL: '.mysqli_connect_error());
}

//printf("Reading data from table: \n");
$res = mysqli_query($conn, 'SELECT * FROM userdetails');
while ($row = mysqli_fetch_assoc($res)) {
$userdetails[] = array($row['username'],$row['userage'],$row['userlat'],$row['userlon']);

}

?>

<script type="text/javascript">

	var username =  <?php echo json_encode($civname); ?>;
	alert(username);


	

	for (i = 0; i < userdetails.length; i++) 
	{
		if(userdetails[i][0]== civname && userdetails[i][1]== civage)
		{
			var update = document.getElementById("update");
			update.innerHTML= 'CIVILIAN LOCATED!';
			var lat = document.getElementById("lat");
			lat.innerHTML= userdetails[i][2];
			var lon = document.getElementById("lon");
			lon.innerHTML= userdetails[i][3];


		}




</script>
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