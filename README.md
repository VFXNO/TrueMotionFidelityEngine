<div align="center">
![TRUE_MOTION_ICON](https://github.com/user-attachments/assets/eae8b0ec-b36b-436d-a3aa-aa897b3dda4c)
</div>
# True Motion Fidelity Engine (TMFE)

<div align="center">

**A high-performance frame generation engine for Windows 11**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%2011-0078D6.svg)]()
[![Build](https://img.shields.io/badge/Build-CMake-064F8C.svg)]()

</div>

---

## 📋 Overview

True Motion Fidelity Engine (TMFE) is an advanced frame generation project that leverages Windows Graphics Capture API, Direct3D 11 compute shaders, and ImGui to deliver smooth motion interpolation on Windows 11. The engine uses shader-based interpolation techniques to generate intermediate frames, enhancing visual fluidity in real-time applications.

### ✨ Key Features

- **Windows Graphics Capture Integration** - Captures screen content with low overhead
- **D3D11 Compute Shader Interpolation** - GPU-accelerated frame generation
- **Real-time GUI** - ImGui-based interface for monitoring and configuration
- **High Performance** - Optimized for minimal latency and maximum throughput
- **Windows 11 Optimized** - Built specifically for Windows 11 graphics stack

---

## 🛠️ Build Instructions

### Prerequisites

Before building TMFE, ensure you have:

- **Windows 11** (required)
- **Visual Studio 2022** with Desktop Development with C++ workload
- **Windows 11 SDK** (version 10.0.26100 or newer recommended)
- **CMake** 3.15 or higher

### Building from Source

1. **Clone the repository**
   ```bash
   git clone https://github.com/VFXNO/TrueMotionFidelityEngine.git
   cd TrueMotionFidelityEngine
   ```

2. **Generate build files**
   ```bash
   cmake -S . -B build -G "Visual Studio 17 2022" -A x64
   ```

3. **Compile the project**
   ```bash
   cmake --build build --config Release
   ```

4. **Run the application**
   ```bash
   build/bin/Release/"True Motion Fidelity Engine.exe"
   ```

### Build Configurations

- **Release** - Optimized build for production use
- **Debug** - Debug symbols and validation layers enabled

---

## 🚀 Usage

After building the project, launch `True Motion Fidelity Engine.exe` from the `build/bin/Release` directory.

### Getting Started

1. Launch the application
2. Use the ImGui interface to configure capture settings
3. Select the target window or display for frame generation
4. Adjust interpolation parameters as needed
5. Monitor performance metrics in real-time

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests (THIS IS VERY IMPORTANT PLEASE IF YOU HAVE SKILLS OR KNOWLIDGE OR EVEN  IF YOU USE AI TO CODE TRY TO PULL REQUEST SO WE COULD IMPROVE IT AND BE FREE FOR ALL ).

### How to Contribute

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License & Branding

Distributed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for more information.

### ⚠️ Important Note on Branding

The name **"True Motion Fidelity Engine"** is a trademark of **VFXNO**.

**You CAN:**
- ✅ Fork this repository and modify the code
- ✅ Use this code commercially
- ✅ Distribute modified versions

**You CANNOT:**
- ❌ Use the name "True Motion Fidelity Engine" or the project's logos in derivative works without permission
- ❌ Imply endorsement by VFXNO without permission

**Requirements:**
- 📝 If you distribute a modified version, you **must rename it**
- 📝 You **must include** the `NOTICE` file in your distribution, keeping the attribution "Powered by VFXNO"

---



<div align="center">

**Powered by VFXNO**

</div>
