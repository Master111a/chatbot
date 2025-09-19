import Image from "next/image";

const WaveBackground = () => {
    return (
        <div className="w-full overflow-hidden">
            <Image
                src="/image.png"
                alt="image"
                width={580}
                height={180}
                className="object-cover w-100 object-center"
            />
        </div>
    );
};

export default WaveBackground;
