const field=document.getElementById('article');
const button=document.getElementById('run');
const counter=document.getElementById('counter');
const status=document.getElementById('status');
field.addEventListener('input',()=>counter.textContent=`${field.value.length} / 10,000 characters`);
button.addEventListener('click',async()=>{
  button.disabled=true;status.textContent='Classifying...';
  try{
    const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:field.value})});
    const data=await response.json();
    if(!response.ok)throw Error(data.error||'Classification failed');
    document.getElementById('empty').hidden=true;
    document.getElementById('result').hidden=false;
    document.getElementById('category').textContent=`${data.label_kin} · ${data.label}`;
    document.getElementById('confidence').textContent=data.top_categories?`Top model scores (not calibrated confidence): ${data.top_categories.map(c=>`${c.label} ${(c.score*100).toFixed(1)}%`).join(' · ')}`:'This model does not output class probabilities.';
    status.textContent='Classification complete.';
  }catch(error){status.textContent=error.message}
  finally{button.disabled=false}
});
