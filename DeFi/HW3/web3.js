async function main () {
    const { Web3 } = require('web3');
const web3 = new Web3('https://mainnet.infura.io/v3/df3803a5e912496284929c8f720b8194'); // Replace with your Ethereum node URL or Infura project ID

const contractAddress = '0x7bAF9BaDDa18F19F175F22bC40e3DeF9461492D7'; // Replace with the address of the smart contract you want to interact with
const storageSlotIndex = 3; // Replace with the index of the storage slot you want to inspect

const contents = await web3.eth.getStorageAt(contractAddress, storageSlotIndex)

console.log(contents)

} 
main().then(() => process.exit(0), e => { console.error(e); process.exit(1) })

;3:7nH|}X;8dxQ3D5(c='rTb9)_B"XJe