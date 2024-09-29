import { useState } from "react"
import { createClient } from '@supabase/supabase-js';
import axios from "axios";
import { serverUrl } from '../utils/api';
import { useNavigate } from "react-router-dom";

const supabase = createClient("https://ujxhafubgfllbkrxxezb.supabase.co", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVqeGhhZnViZ2ZsbGJrcnh4ZXpiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjczMzA2ODgsImV4cCI6MjA0MjkwNjY4OH0.HedtgWuXLq7UHI2xR196dGed3JoV-FHjHg5nhJN7lLA");

const VideoCheck = () => {
  const navigate  = useNavigate();
  const [title, setTitle] = useState<string>("");
  const [progress1, setProgress1] = useState(0);
  const [selectedFile1, setSelectedFile1] = useState<any>();
  const [videoUrl, setVideoUrl] = useState<string>("");
  const [is_submit_valid, set_is_submit_valid] = useState<Boolean|null>(false);



  const handleTitleChange = (e: any) => {
    setTitle(e.target.value);
  }

  
  const handleFileInput1 = (e: any) => {
    setSelectedFile1(e.target.files[0]);
  }

  const uploadFile1 = async (file: File, bucket: string, setProgress: React.Dispatch<React.SetStateAction<number>>, setUrl: React.Dispatch<React.SetStateAction<string>>) => {
    if (!file) return;

    const fileExt = file.name.split('.').pop();
    const fileName = `${Math.random()}.${fileExt}`;
    const filePath = `${fileName}+${file.name}`;
    console.log("video tak aa gye hai")

    try {
        const { data, error } = await supabase.storage
            .from(bucket)
            .upload(filePath, file, {
                cacheControl: '3600',
                upsert: false
            });
        console.log("yaha tak to sahi hai")
        if (error) throw error;

        const { data: { publicUrl } } = supabase.storage
            .from(bucket)
            .getPublicUrl(filePath);
        console.log(data);
        setUrl(publicUrl);
        set_is_submit_valid(true);
        setProgress(100);

    } catch (error) {
        console.error('Error uploading file:', error);
        setProgress(0);
    }
}

const handleFinalSubmit = async (e: any) => {
  e.preventDefault();
  var storeStruct = {
      name: title,
      videoUrl: videoUrl
  }
  const response = await axios.post(`${serverUrl}/test_video_url`, storeStruct)
  if (response.status === 200) {
      setProgress1(0)
      setVideoUrl("")
      setTitle("")
      setSelectedFile1(null)
      set_is_submit_valid(false)
  }
}


const handle_backToHome = async () =>{
  const deleteThisVideo = {
    name:title
  }
  const response = await axios.post(`${serverUrl}/delete_video`,deleteThisVideo)
  if (response.status === 200){
    navigate('/')
  }
}

  return (<>
    <div>What should we call you...</div>
    <input type="text" value={title} onChange={handleTitleChange} />
    <br></br>
    <div>Upload your video here : {progress1}%</div>
    <input type="file" onChange={handleFileInput1} />
    <button onClick={() => uploadFile1(selectedFile1, 'videos_to_check', setProgress1, setVideoUrl)}> Upload to S3</button>
    <br></br>
    <div>
      <button disabled={!is_submit_valid} onClick={handleFinalSubmit}>Check</button>
    </div>
    <div>Report is here</div>
    <button onClick={handle_backToHome}>Back to Home</button>
  </>)
}

export default VideoCheck