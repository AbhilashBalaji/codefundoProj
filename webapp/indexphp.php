<!DOCTYPE html>
<html>
<head>
	<title>HURRICANE WARNING SYSTEM</title>
	<script type="text/javascript" src="//code.jquery.com/jquery-2.1.0.min.js"></script>

	
</head>
<body>

	
	<button name="safelocationmapbutton" onclick="sendsever()">GET USER LOCATION</button>
	<a href="civilianmap.php">click here to load map</a>

	<form name="myform" action="civilianserver.php" method="GET" target="_blank">
	<input type="hidden" name="latitude" id="latitude" />
	<input type="hidden" name="longitude" id="longitude" />
	<input type="submit">
	</form>

	<script type="text/javascript">
		
    
    function sendsever() 
    {         
        if (navigator.geolocation) 
        {
            navigator.geolocation.getCurrentPosition(showPosition);
        } 

    }
    function showPosition(position) 
    {
      
        var userlat;
        var userlon;
        userlat = (position.coords.latitude);  
        userlon = (position.coords.longitude); 
        alert("postiion recieved")   ;

        document.getElementById("latitude").value = userlat ;
        document.getElementById("longitude").value = userlon ;
        alert(document.getElementById("latitude").value);
       

     //window.location.href= "civilianserver.php";

    }

    

	</script>
</body>
</html>