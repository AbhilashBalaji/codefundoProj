
<!DOCTYPE html>
<html>
  <head>
    <meta name="viewport" content="initial-scale=1.0, user-scalable=no">
    <meta charset="utf-8">
    <title>Directions Service</title>
    <link rel="stylesheet" type="text/css" href="civilianmap.css">
    <script type="text/javascript" src="//code.jquery.com/jquery-2.1.0.min.js"></script>
  

  </head>
  <body>
  		
  
    

    <div id="floating-panel">
    <b>Start: </b>
    <select id="start" style="width: 150px; height: 30px">
      <option value="12.2958,76.6394">YOUR LOCATION</option>
      <option value="12.2958,76.6394">TEST</option>

    </select>
    <b>End: </b>
    <select id="end" style="width: 150px; height: 30px">
      <option value="12.9716,77.5946" id="safeloc1">LOCATION 1</option>
      <option value="12.2958,76.6394" id="safeloc2">LOCATION 2</option>
      <option value="12.2958,76.6394" id="safeloc3">LOCATION 3</option>
      <option value="12.2958,76.6394" id="safeloc4">LOCATION 4</option>
      <option value="19.0760,72.8777" id="safeloc5">LOCATION 5</option>
    </select>
    </div>
    <div id="map"></div>


    <?php


$userlat = $_GET['latitude']; 
$userlon = $_GET['longitude'];
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
$res = mysqli_query($conn, 'SELECT * FROM safelocations');
while ($row = mysqli_fetch_assoc($res)) {
$safelat[] = ($row['safelat']);
$safelon[] = ($row['safelon']);
}



?>





    <script>

      ////script to get user location 

    
      var userlat = "<?php echo $userlat ?>"
      var userlon = "<?php echo $userlon ?>"
 



      
    ////////////////////////////////////////////////////////////////////////

    ///GETTING ALL THE LAT AND LON VARIABLES FROM THE SEVER
    var safelat =  <?php echo json_encode($safelat); ?>;
    var safelon =  <?php echo json_encode($safelon); ?>;

    document.getElementById('safeloc1').options[0].value = safelat[0] + ","+safelon[0];
    document.getElementById('safeloc2').options[0].value = safelat[1] + ","+safelon[1];
    document.getElementById('safeloc3').options[0].value = safelat[2] + ","+safelon[2];
    document.getElementById('safeloc4').options[0].value = safelat[3] + ","+safelon[3];
    document.getElementById('safeloc5').options[0].value = safelat[4] + ","+safelon[4];
    
     document.getElementById('start').options[0].value = userlat + ","+userlon;

     alert(document.getElementById('safeloc1').options[0].value);

      function initMap() {
        var directionsService = new google.maps.DirectionsService;
        var directionsDisplay = new google.maps.DirectionsRenderer;
        var map = new google.maps.Map(document.getElementById('map'), {
          zoom: 13,
          //center: {lat: userlat, lng: userlon}
          center: {lat: 12.2958, lng: 76.6394}
        });
        directionsDisplay.setMap(map);

        var onChangeHandler = function() {
          calculateAndDisplayRoute(directionsService, directionsDisplay);
        };
        
        document.getElementById('start').addEventListener('change', onChangeHandler);
        document.getElementById('end').addEventListener('change', onChangeHandler);
      }

      
      function calculateAndDisplayRoute(directionsService, directionsDisplay) {

        directionsService.route({
          origin: document.getElementById('start').value,
          destination: document.getElementById('end').value,
          travelMode: 'DRIVING'
        }, function(response, status) {
          if (status === 'OK') {
            directionsDisplay.setDirections(response);
          } else {
            window.alert('Directions request failed due to ' + status);
          }
        });
      }
    </script>
    <script async defer
    src="https://maps.googleapis.com/maps/api/js?key=AIzaSyD8ddNl01F64wExwiDBRSIFjiAHXnT4_uQ&callback=initMap">
    </script>



  </body>
</html>