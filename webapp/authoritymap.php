<!DOCTYPE html>
<html>
<head>
	<title>LOCATE CIVILIANS</title>
	<meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<link href="style.css" rel="stylesheet">
</head>
<body>



<?php
//fetching data from server
$host = 'civilianserver.mysql.database.azure.com';
$username = 'serveradmin@civilianserver';
$password = 'itsnotpesu123!';
$db_name = 'civiliandb';


$userlat = array();
$userlon = array();
//Establishes the connection
$conn = mysqli_init();
mysqli_real_connect($conn, $host, $username, $password, $db_name, 3306);
if (mysqli_connect_errno($conn)) {
die('Failed to connect to MySQL: '.mysqli_connect_error());
}

//printf("Reading data from table: \n");
$res = mysqli_query($conn, 'SELECT * FROM helpusers');
while ($row = mysqli_fetch_assoc($res)) {
$userlat[] = ($row['userlat']);
$userlon[] = ($row['userlon']);
}

?>

<div id="map"></div>


<script type="text/javascript">
    
    var userlat =  <?php echo json_encode($userlat); ?>;
    var userlon =  <?php echo json_encode($userlon); ?>;

</script>
<script>
	


function initMap() {

	var map = new google.maps.Map(document.getElementById('map'), {
		zoom: 13,
		center: new google.maps.LatLng(12.923455, 77.568051),
		mapTypeId: google.maps.MapTypeId.ROADMAP
	});

	var infowindow = new google.maps.InfoWindow({});

	var marker, i;

	for (i = 0; i < userlat.length; i++) {
		marker = new google.maps.Marker({
			position: new google.maps.LatLng(userlat[i], userlon[i]),
			map: map
		});

		google.maps.event.addListener(marker, 'click', (function (marker, i) {
			return function () {
				//infowindow.setContent(locations[i][0]);
				infowindow.open(map, marker);
			}
		})(marker, i));
	}
}

</script>
	<script async defer 
	src="https://maps.googleapis.com/maps/api/js?key=AIzaSyCDanZHG6pHmwIgapfFqx77F14I65ahUzM&callback=initMap"></script>


</body>
</html>