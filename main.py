from decoder import Decoder

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    decoder = Decoder('sample_data/example2.wav')
    decoder.decode(show_img=True, outfile="Result.jpg")
