var num_greenhouses = 1

// get the container element
const container = document.querySelector(".container");

  var num_greenhouses_field = document.getElementById("num_greenhouses")
  var next_num_greenhouses = 1;
  num_greenhouses_field.addEventListener('input', function (evt) {
    console.log("hjabdhabed")
    console.log(this.value)
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
    
    const minTempInput = createInfoSpan(`min-temp-${i}`, "GreenHouseTemp");
    const maxTempInput = createInfoSpan(`max-temp-${i}`, "OutsideTemp");
    const TimeInput = createInfoSpan(`Time-${i}`, "Time");
    const Heating_Status= createInfoSpan(`HeatingStatus-${i}`, "HeatingStatus");
    const Cooling_status= createInfoSpan(`CoolingStatus-${i}`, "CoolingStatus");
    const Average_consumption= createInfoSpan(`AverageConsumption-${i}`, "AverageConsumption");
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
    
    fetch(`../json/gh${i + 1}_settings.json`)
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
        console.log("sexd");
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
  fetch("./json/gh1_obs.json")
    .then(response => response.json())
    .then(data => {
      const roomData = data[0]; // get the first object in the array
      const minTempInput = document.getElementById("min-temp-0");
      const maxTempInput = document.getElementById("max-temp-0");
      const TimetInput = document.getElementById("Time-0");
      const CoolingStatusInput = document.getElementById("CoolingStatus-0");
      const Heating_StatusInput = document.getElementById("HeatingStatus-0");
      const Average_consumptionInput = document.getElementById("AverageConsumption-0");
      function boldHTML(text) {
        var element = document.createElement("b");
        element.innerHTML = text;
        return element;
      }
      // displaying environment observations
      while( minTempInput.firstChild ) {
        minTempInput.removeChild( minTempInput.firstChild );
      }
      minTempInput.appendChild(boldHTML("I Temp: "))
      minTempInput.appendChild( document.createTextNode(roomData.Greenhouse_temp.toFixed(1)) );

      while( maxTempInput.firstChild ) {
        maxTempInput.removeChild( maxTempInput.firstChild );
      }
      maxTempInput.appendChild(boldHTML("O Temp: "))
      maxTempInput.appendChild( document.createTextNode(roomData.Outside_temp.toFixed(1)) );
      

      while( TimetInput.firstChild ) {
        TimetInput.removeChild( TimetInput.firstChild );
      }
      TimetInput.appendChild(boldHTML("Time: "))
      TimetInput.appendChild( document.createTextNode(roomData.Time) );
      

      while( CoolingStatusInput.firstChild ) {
        CoolingStatusInput.removeChild( CoolingStatusInput.firstChild );
      }
      CoolingStatusInput.appendChild(boldHTML("Vent: "))
      CoolingStatusInput.appendChild( document.createTextNode(roomData.Ventilation) );

      while( Heating_StatusInput.firstChild ) {
        Heating_StatusInput.removeChild( Heating_StatusInput.firstChild );
      }
      Heating_StatusInput.appendChild(boldHTML("Heat: "))
      Heating_StatusInput.appendChild( document.createTextNode(roomData.Heating) );

      while( Average_consumptionInput.firstChild ) {
        Average_consumptionInput.removeChild( Average_consumptionInput.firstChild );
      }
      Average_consumptionInput.appendChild(boldHTML("Avg Energy: "))
      Average_consumptionInput.appendChild( document.createTextNode(roomData.Average_consumption.toFixed()) );


    })
    .catch(error => console.error(error));  
}
setInterval(fetchData, 100);

document.addEventListener("DOMContentLoaded", function() { 
  create_greenhouses();
  /*
  for (let i = 1; i <= num_greenhouses; i++) {
    fetch(`./room${i}.json`)
      .then((response) => response.json())
      .then((data) => {
        // display the data to the user
        console.log(`Greenhouse ${i}:`);
        console.log(data);

        document.getElementById(`room-${i}-min`).value = data.minTemp;
        document.getElementById(`room-${i}-max`).value = data.maxTemp;
        document.getElementById(`room-${i}-crit-min`).value = data.critMinTemp;
        document.getElementById(`room-${i}-crit-max`).value = data.critMaxTemp;
        document.getElementById(`room-${i}-max-time`).value = data.maxTime;
        document.getElementById(`room-${i}-rate`).value = data.rateOfChange;
      })
      .catch((error) => {
        console.error(`Failed to load room ${i}`);
      });
  }*/
});


function submitData() {
  const inputs = document.querySelectorAll(".room-input");
  const allFilled = Array.from(inputs).every((input) => input.value !== "");
  if (allFilled) {

    const roomData = {
      rooms: [],
    };
    console.log("num greenhouses")
    console.log(num_greenhouses)
    console.log(next_num_greenhouses)
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

    num_greenhouses = next_num_greenhouses;

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
          
        } else {
          console.error("Failed to store data");
        }
      })
      .catch((error) => {
        console.log("Error storing data");
      });
      create_greenhouses();
  }
  else{
    console.log("fill all");
  }
}
//const submitButton = document.createElement("button");
//submitButton.textContent = "Submit";
const submit_button = document.getElementById("submit");
submit_button.addEventListener("click", submitData);
//container.appendChild(submitButton);