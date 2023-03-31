// ENABLE DATABASE LOGGING HERE
var h_logging = false;

var num_greenhouses = 1;

// get the container element
const container = document.querySelector(".container");

var num_greenhouses_field = document.getElementById("num_greenhouses")
var next_num_greenhouses = 1;
num_greenhouses_field.addEventListener('input', function (evt) {
  if(this.value>0 || this.value !== null){
    next_num_greenhouses = this.value;
  }
  else{
    next_num_greenhouses = 1;
  }
  
});

// Create rooms
function create_greenhouses(){
  
  container.innerHTML = '';
  for (let i = 0; i < num_greenhouses; i++) {

    const greenhouseContainer = document.createElement("div");
    greenhouseContainer.classList.add("greenhouse_container");
    container.appendChild(greenhouseContainer);
    
    const minTempInput = createInfoSpan(`greenhouse-temp-${i+1}`, "GreenHouseTemp");
    const maxTempInput = createInfoSpan(`outside-temp-${i+1}`, "OutsideTemp");
    const TimeInput = createInfoSpan(`Time-${i+1}`, "Time");
    const Heating_Status= createInfoSpan(`HeatingStatus-${i+1}`, "HeatingStatus");
    const Cooling_status= createInfoSpan(`CoolingStatus-${i+1}`, "CoolingStatus");
    const Average_consumption= createInfoSpan(`AverageConsumption-${i+1}`, "AverageConsumption");
    const roomRow = document.createElement("div");
    roomRow.classList.add("room-row");
  
    roomRow.appendChild(minTempInput);
    roomRow.appendChild(maxTempInput);
    roomRow.appendChild(TimeInput);
    roomRow.appendChild(Heating_Status);
    roomRow.appendChild(Cooling_status);
    roomRow.appendChild(Average_consumption);
  
    const infoContainer = document.createElement("div");
    infoContainer.classList.add("info_container");
    const roomLabel = document.createElement("H1");
    roomLabel.innerHTML = `Greenhouse ${i + 1}`;
    greenhouseContainer.appendChild(roomLabel);
    infoContainer.appendChild(roomRow);
  
    greenhouseContainer.appendChild(infoContainer);
  
    createInputArea(i, greenhouseContainer);
    
    fetch(`./json/gh${i + 1}_settings.json`)
      .then(response => response.json())
      .then(data => {
        
        document.getElementById(`room-${i + 1}-min`).value = data.minTemp;
        document.getElementById(`room-${i + 1}-max`).value = data.maxTemp;
        document.getElementById(`room-${i + 1}-crit-min`).value = data.critMinTemp;
        document.getElementById(`room-${i + 1}-crit-max`).value = data.critMaxTemp;
        document.getElementById(`room-${i + 1}-max-time`).value = data.maxTime;
        document.getElementById(`room-${i + 1}-rate`).value = data.rateOfChange;
        
      })
      .catch(error => {
        console.error(error); 
      });
  }
}
function createInfoSpan(id) {
  const info_span = document.createElement("SPAN");
  info_span.id = id;
  return info_span;
}
function createInputArea(index, greenhouseContainer) {
  const roomContainer = document.createElement("div");
  roomContainer.classList.add("input_container");

  const roomLabel = document.createElement("label");
  roomContainer.appendChild(roomLabel);

  // create inputs and corresponding labels
  var temp;
  temp = createInputAndLabel("Min Temp:", `room-${index + 1}-min`);
  const minTempInput = temp[0];
  const minTempInputLabel = temp[1];
  temp = createInputAndLabel("Max Temp:", `room-${index + 1}-max`);
  const maxTempInput = temp[0];
  const maxTempInputLabel = temp[1];
  temp = createInputAndLabel("Crit. Min Temp:", `room-${index + 1}-crit-min`);
  const critMinTempInput = temp[0];
  const critMinTempInputLabel = temp[1];
  temp = createInputAndLabel("Crit. Max Temp:", `room-${index + 1}-crit-max`);
  const critMaxTempInput = temp[0];
  const critMaxTempInputLabel = temp[1];
  temp = createInputAndLabel("Crit. Time:", `room-${index + 1}-max-time`);
  const maxTimeInput = temp[0];
  const maxTimeInputLabel = temp[1];
  temp = createInputAndLabel("Rate of Change:", `room-${index + 1}-rate`);
  const rateInput = temp[0];
  const rateInputLabel = temp[1];

  const roomRow = document.createElement("div");
  roomRow.classList.add("room-row");
  var div0 = document.createElement("div");
  div0.appendChild(minTempInputLabel)
  div0.appendChild(minTempInput)
  roomRow.appendChild(div0);

  var div1 = document.createElement("div");
  div1.appendChild(maxTempInputLabel);
  div1.appendChild(maxTempInput);
  roomRow.appendChild(div1);

  var div2 = document.createElement("div");
  div2.appendChild(critMinTempInputLabel);
  div2.appendChild(critMinTempInput);
  roomRow.appendChild(div2);

  var div3 = document.createElement("div");
  div3.appendChild(critMaxTempInputLabel);
  div3.appendChild(critMaxTempInput);
  roomRow.appendChild(div3);

  var div4 = document.createElement("div");
  div4.appendChild(maxTimeInputLabel);
  div4.appendChild(maxTimeInput);
  roomRow.appendChild(div4);

  var div5 = document.createElement("div");
  div5.appendChild(rateInputLabel);
  div5.appendChild(rateInput);
  roomRow.appendChild(div5);

  roomContainer.appendChild(roomRow);

  greenhouseContainer.appendChild(roomContainer);
}

function createInputAndLabel(labelText,id,) {
  const input = document.createElement("input");
  const label = document.createElement("Label");
  input.setAttribute("type", "number");
  input.setAttribute("id",id);
  label.htmlFor = id;
  label.innerHTML=labelText;
  input.classList.add("room-input");
  return [input, label];
}
// Fetch data from data.json
function fetchData() {
  for (let i = 0; i < num_greenhouses; i++) {
    
    fetch(`./json/gh${i + 1}_obs.json`)
      .then(response => response.json())
      .then(data => {
        const roomData = data; // get the first object in the array
        
        const outsideTempInput = document.getElementById(`greenhouse-temp-${i+1}`);
        const insideTempInput = document.getElementById(`outside-temp-${i+1}`);
        const TimetInput = document.getElementById(`Time-${i+1}`);
        const CoolingStatusInput = document.getElementById(`CoolingStatus-${i+1}`);
        const Heating_StatusInput = document.getElementById(`HeatingStatus-${i+1}`);
        const Average_consumptionInput = document.getElementById(`AverageConsumption-${i+1}`);

        // displaying environment observations
        outsideTempInput.innerHTML = '';
        var span1 = document.createElement('span');
        var span2 = document.createElement('span');
        outsideTempInput.appendChild(span1);
        outsideTempInput.appendChild(span2);
        outsideTempInput.children[0].innerHTML =  "I Temp: ";
        outsideTempInput.children[1].innerHTML =  roomData.Greenhouse_temp.toFixed(1);

        insideTempInput.innerHTML = '';
        span1 = document.createElement('span');
        span2 = document.createElement('span');
        insideTempInput.appendChild(span1);
        insideTempInput.appendChild(span2);
        insideTempInput.children[0].innerHTML =  "O Temp: ";
        insideTempInput.children[1].innerHTML =  roomData.Outside_temp.toFixed(1);

        TimetInput.innerHTML = '';
        span1 = document.createElement('span');
        span2 = document.createElement('span');
        TimetInput.appendChild(span1);
        TimetInput.appendChild(span2);
        TimetInput.children[0].innerHTML =  "Time (d:h:m): ";
        TimetInput.children[1].innerHTML =  roomData.Time;
        
        if (roomData.Ventilation == 0){
          var temp = "Off";
        }
        else{
          var temp = "On"
        }
        CoolingStatusInput.innerHTML = '';
        span1 = document.createElement('span');
        span2 = document.createElement('span');
        CoolingStatusInput.appendChild(span1);
        CoolingStatusInput.appendChild(span2);
        CoolingStatusInput.children[0].innerHTML =  "Vent: ";
        CoolingStatusInput.children[1].innerHTML =  temp;
        
        if (roomData.Heating == 0){
          var temp = "Off";
        }
        else{
          var temp = "On"
        }
        Heating_StatusInput.innerHTML = '';
        span1 = document.createElement('span');
        span2 = document.createElement('span');
        Heating_StatusInput.appendChild(span1);
        Heating_StatusInput.appendChild(span2);
        Heating_StatusInput.children[0].innerHTML =  "Heat: ";
        Heating_StatusInput.children[1].innerHTML =  temp;
        
        if (roomData.Average_consumption == -5){
          var temp = "-"
        }
        else{
          var temp = roomData.Average_consumption*2.7;
          temp = temp.toFixed(2);
        }
        Average_consumptionInput.innerHTML = '';
        span1 = document.createElement('span');
        span2 = document.createElement('span');
        Average_consumptionInput.appendChild(span1);
        Average_consumptionInput.appendChild(span2);
        Average_consumptionInput.children[0].innerHTML =  "Avg Energy/H (kW): ";
        Average_consumptionInput.children[1].innerHTML =  temp;
      })
      .catch(error => console.error(error)); 
  }
  
}
setInterval(fetchData, 1000);

function pushData(){
  var env_data = {
    env_obs: []
  };
  for (let i = 0; i < num_greenhouses; i++) {
    //console.log(env_data);
    const greenhouse_temp = document.getElementById(`greenhouse-temp-${i+1}`);
    const outside_temp = document.getElementById(`outside-temp-${i+1}`);
    const time = document.getElementById(`Time-${i+1}`);
    const vent = document.getElementById(`CoolingStatus-${i+1}`);
    const heat = document.getElementById(`HeatingStatus-${i+1}`);
    const avg_number = document.getElementById(`AverageConsumption-${i+1}`);
    //const jsonData = env_data;
    const obs = {
      greenhouse_temp: greenhouse_temp.children[1].innerHTML,
      outside_temp: outside_temp.children[1].innerHTML,
      time: time.children[1].innerHTML,
      vent: vent.children[1].innerHTML,
      heat: heat.children[1].innerHTML,
      avg_number: avg_number.children[1].innerHTML,
    };
    env_data.env_obs.push(obs);
    
  }
  const jsonData = JSON.stringify(env_data);
  
  fetch("./write_obs_to_db", {
      
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: jsonData,
  })
    .then((response) => {
      if (response.ok) {
        console.log("Data stored successfully");
        
      } else {
        console.error("Failed to store data");
      }
    })
    .catch((error) => {
      console.log("Error storing data");
    });
}
if (h_logging){
  setInterval(pushData, 30000);
}

function submitData() {
  const inputs = document.querySelectorAll(".room-input");
  const allFilled = Array.from(inputs).every((input) => input.value !== "");
  if (allFilled) {

    const roomData = {
      greenhouse_nums: [],
      rooms: [],
      location: [],
    };
    const numData = {
      prev_num: num_greenhouses,
      current_num: next_num_greenhouses,
    };

    roomData.greenhouse_nums.push(numData);

    const location_input = document.getElementById("location_input");
    const location = {
      location: location_input.value,
    }

    roomData.location.push(location);
    
    for (let i = 1; i <= num_greenhouses; i++) {
      
      const minTempInput = document.getElementById(`room-${i}-min`);
      const maxTempInput = document.getElementById(`room-${i}-max`);
      const critMinTempInput = document.getElementById(`room-${i}-crit-min`);
      const critMaxTempInput = document.getElementById(`room-${i}-crit-max`);
      const maxTimeInput = document.getElementById(`room-${i}-max-time`);
      const rateInput = document.getElementById(`room-${i}-rate`);

      const room = {
        minTemp: minTempInput.value,
        maxTemp: maxTempInput.value,
        critMinTemp: critMinTempInput.value,
        critMaxTemp: critMaxTempInput.value,
        maxTime: maxTimeInput.value,
        rateOfChange: rateInput.value,
      };
      roomData.rooms.push(room);
      
    }
    const jsonData = JSON.stringify(roomData);
    
    fetch("./write_to_json", {
      
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: jsonData,
    })
      .then((response) => {
        if (response.ok) {
          console.log("Data stored successfully");
          num_greenhouses = next_num_greenhouses;
          create_greenhouses();
          
        } else {
          console.error("Failed to store data");
        }
      })
      .catch((error) => {
        console.log("Error storing data");
      });
      
  }
  else{
    alert("Please fill all values.");
  }
}

const submit_button = document.getElementById("submit");
submit_button.addEventListener("click", submitData);

document.addEventListener("DOMContentLoaded", function() { 
  var num_greenhouses_field = document.getElementById("num_greenhouses")
  num_greenhouses_field.value = 1;
  create_greenhouses();
});