import whisper
import librosa
import warnings
warnings.filterwarnings("ignore")

def transcribe_with_librosa(file_path, output_txt="transcript.txt"):
    print("جاري قراءة الملف الصوتي...")
    audio_data, sr = librosa.load(file_path, sr=16000)
    
    # تحميل نموذج small
    model = whisper.load_model("small")
    
    print("جاري التفريغ بدقة عالية وتوجيه إملائي للقرآن الكريم...")
    
    # التوجيه المسبق لمنع أخطاء الحروف المقطعة والآيات
    prompt_text = "طه ما أنزلنا عليك القرآن لتشقى إلا تذكرة لمن يخشى"
    result = model.transcribe(audio_data, language="ar", initial_prompt=prompt_text)
    
    print("\n--- النص المفرغ ---")
    print(result["text"])
    
    # حفظ النص في الملف
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(result["text"])
        
    print(f"\nتم حفظ النص بنجاح في الملف: {output_txt}")

if __name__ == "__main__":
    transcribe_with_librosa(r"D:\lecture_transcriber\1.wav.wav")