<template>
  <div id="app">
    <nav>
      <h1>电话本管理系统</h1>
    </nav>
    <router-view />
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style>
#app {
  font-family: Arial, sans-serif;
  text-align: center;
}

nav {
  background-color: #42b983;
  padding: 10px;
  color: white;
}

main {
  margin: 20px;
}
</style>
